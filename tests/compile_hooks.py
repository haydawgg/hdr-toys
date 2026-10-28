"""Compile the changed mpv hooks as Vulkan GLSL.

The prelude follows libplacebo's hook bindings: parameters live in one
uniform block, //!BUFFER members live in storage blocks, and bound
textures expose the *_raw, *_mul, and *_tex names those passes use.
glslangValidator emits SPIR-V. D3D11 is libplacebo's later HLSL step
and is not run here.

    py -3.14 tests/compile_hooks.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADERS = [
    ROOT / "shaders" / "hdr-toys" / "gamut-mapping" / "bottosson.glsl",
    ROOT / "shaders" / "hdr-toys" / "tone-mapping" / "astra.glsl",
]
CHANGED = {
    "gamut mapping (bottosson, boundary generation)",
    "gamut mapping (bottosson, boundary cache commit)",
    "gamut mapping (bottosson)",
    "tone mapping (astra, lut generation)",
    "tone mapping (astra, chroma boundary)",
    "tone mapping (astra, lut cache commit)",
}


def fail(message):
    raise SystemExit(message)


def split_hooks(text):
    hooks = []
    parts = re.split(r"(?=^//!HOOK\b)", text, flags=re.M)
    for part in parts:
        if not part.startswith("//!HOOK"):
            continue
        lines = part.splitlines(keepends=True)
        body_start = 0
        for index, line in enumerate(lines):
            if line.startswith("//!"):
                body_start = index + 1
            elif line.strip() == "":
                continue
            else:
                break
        header_text = "".join(lines[:body_start])
        body_text = "".join(lines[body_start:])
        desc = re.search(r"//!DESC\s+(.+)", header_text)
        compute = re.search(r"//!COMPUTE\s+(\d+)\s+(\d+)", header_text)
        binds = re.findall(r"//!BIND\s+(\w+)", header_text)
        hooks.append({
            "desc": desc.group(1).strip() if desc else "unnamed",
            "compute": (
                (int(compute.group(1)), int(compute.group(2)))
                if compute else None
            ),
            "binds": binds,
            "body": body_text,
        })
    return hooks


def parse_resources(text):
    """Return (params, defines, buffers) declared before the hook bodies."""
    params = []
    defines = []
    buffers = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("//!HOOK"):
            current = None
            continue
        if line.startswith("//!PARAM"):
            current = {"name": line.split()[1], "kind": None, "initial": None}
            continue
        if current is not None and "kind" in current and "vars" not in current:
            if line.startswith("//!TYPE"):
                parts = line.split()
                current["kind"] = "DEFINE" if "DEFINE" in parts else parts[-1]
                continue
            if line.startswith("//!") or line == "" or line.startswith("//"):
                continue
            current["initial"] = line.split()[0]
            if current["kind"] == "DEFINE":
                defines.append((current["name"], current["initial"]))
            elif current["kind"] in {"float", "int", "uint"}:
                params.append((current["kind"], current["name"]))
            else:
                fail(f"unsupported parameter type for {current['name']}")
            current = None
            continue
        if line.startswith("//!BUFFER"):
            current = {
                "name": line.split()[1],
                "storage": False,
                "vars": [],
            }
            buffers.append(current)
            continue
        if line.startswith("//!STORAGE") and current and "vars" in current:
            current["storage"] = True
            continue
        if line.startswith("//!VAR") and current and "vars" in current:
            rest = line[len("//!VAR"):].strip().rstrip(";")
            type_name, var_name = rest.split(None, 1)
            current["vars"].append((type_name, var_name))
            continue
    return params, defines, buffers


def prelude(hook, params, defines, buffers):
    lines = ["#version 450", ""]
    for name, initial in defines:
        lines.append(f"#define {name} {initial}")
    if defines:
        lines.append("")

    binding = 0
    if params:
        lines.append(
            "layout(std140, set = 0, binding = "
            f"{binding}) uniform HookParams {{"
        )
        binding += 1
        for kind, name in params:
            lines.append(f"    {kind} {name};")
        lines.append("};")
        lines.append("")

    for buf in buffers:
        layout = "std430" if buf["storage"] else "std140"
        qualifier = "coherent buffer" if buf["storage"] else "uniform"
        lines.append(
            f"layout({layout}, set = 0, binding = {binding}) "
            f"{qualifier} {buf['name']} {{"
        )
        binding += 1
        for type_name, var_name in buf["vars"]:
            lines.append(f"    {type_name} {var_name};")
        lines.append("};")
        lines.append("")

    for name in hook["binds"]:
        lines.append(
            f"layout(set = 0, binding = {binding}) uniform sampler2D {name}_raw;"
        )
        binding += 1
        lines.append(f"#define {name}_pos vec2(0.5)")
        lines.append(f"#define {name}_size vec2(1920.0, 1080.0)")
        lines.append(f"#define {name}_pt vec2(1.0 / 1920.0, 1.0 / 1080.0)")
        lines.append(f"#define {name}_off vec2(0.0)")
        lines.append(f"#define {name}_mul 1.0")
        lines.append(f"#define {name}_rot mat2(1.0, 0.0, 0.0, 1.0)")
        lines.append(f"#define {name}_map {name}_pos")
        lines.append(
            f"#define {name}_tex(pos) ({name}_mul * vec4(textureLod({name}_raw, pos, 0.0)))"
        )
        lines.append(
            f"#define {name}_texOff(off) ({name}_tex({name}_pos + {name}_pt * vec2(off)))"
        )
    if hook["binds"]:
        lines.append("")

    if hook["compute"]:
        sx, sy = hook["compute"]
        lines.append(
            f"layout(local_size_x = {sx}, local_size_y = {sy}, local_size_z = 1) in;"
        )
        lines.append(
            "layout(rgba16f, set = 0, binding = "
            f"{binding}) uniform writeonly image2D out_image;"
        )
    else:
        lines.append("layout(location = 0) out vec4 hook_output;")
    lines.append("")
    return "\n".join(lines)


def epilogue(hook):
    if hook["compute"]:
        return "\nvoid main() { hook(); }\n"
    return "\nvoid main() { hook_output = hook(); }\n"


def find_validator():
    found = shutil.which("glslangValidator") or shutil.which("glslang")
    if found:
        return found
    root = Path(os.environ.get("TEMP", ".")) / "glslang" / "bin"
    for name in ("glslangValidator.exe", "glslang.exe"):
        candidate = root / name
        if candidate.exists():
            return str(candidate)
    return None


def main():
    validator = find_validator()
    if validator is None:
        fail("glslangValidator was not found")

    compiled = 0
    with tempfile.TemporaryDirectory(prefix="hdr-toys-glsl-") as directory:
        for path in SHADERS:
            text = path.read_text(encoding="utf-8")
            params, defines, buffers = parse_resources(text)
            for hook in split_hooks(text):
                if hook["desc"] not in CHANGED:
                    continue
                stage = "comp" if hook["compute"] else "frag"
                source_path = Path(directory) / f"{compiled}.glsl"
                source_path.write_text(
                    prelude(hook, params, defines, buffers)
                    + hook["body"]
                    + epilogue(hook),
                    encoding="utf-8",
                )
                output = source_path.with_suffix(".spv")
                result = subprocess.run(
                    [
                        validator,
                        "-V",
                        "--target-env", "vulkan1.3",
                        "-S", stage,
                        "-o", str(output),
                        str(source_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(result.stdout)
                    print(result.stderr)
                    kept = Path(os.environ.get("TEMP", ".")) / f"hdr-toys-failed-{compiled}.glsl"
                    kept.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
                    print(f"source kept at {kept}")
                    fail(f"Vulkan compile failed: {hook['desc']}")
                print(f"vulkan spir-v: {hook['desc']}")
                compiled += 1
    if compiled != len(CHANGED):
        fail(f"compiled {compiled} passes, expected {len(CHANGED)}")
    print(f"compiled {compiled} Vulkan passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
