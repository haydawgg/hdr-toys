"""Numeric checks for the HDR Toys performance and picture plan.

These tests mirror the shader math. They do not replace a Vulkan frame
timer. Run from the repository root:

    py -3.14 tests/test_hdr_plan.py
"""

import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASTRA = ROOT / "shaders" / "hdr-toys" / "tone-mapping" / "astra.glsl"
BOTTOSSON = ROOT / "shaders" / "hdr-toys" / "gamut-mapping" / "bottosson.glsl"

BT2020_TO_XYZ = np.array([
    [0.6369580483012914, 0.1446169035862083, 0.1688809751641721],
    [0.2627002120112671, 0.6779980715188708, 0.0593017164698620],
    [0.0000000000000000, 0.0280726930490874, 1.0609850577107910],
], dtype=np.float64)

XYZ_TO_BT709 = np.array([
    [3.2409699419045226, -1.5373831775700940, -0.4986107602930034],
    [-0.9692436362808796, 1.8759675015077202, 0.0415550574071756],
    [0.0556300796969937, -0.2039769588889765, 1.0569715142428786],
], dtype=np.float64)

OKLAB_TO_LMS = np.array([
    [1.0000000000000000, 0.3963377773761749, 0.2158037573099136],
    [1.0000000000000000, -0.1055613458156586, -0.0638541728258133],
    [1.0000000000000000, -0.0894841775298119, -1.2914855480194092],
], dtype=np.float64)

LMS_TO_XYZ = np.array([
    [1.2268798758459243, -0.5578149944602171, 0.2813910456659647],
    [-0.0405757452148008, 1.1122868032803170, -0.0717110580655164],
    [-0.0763729366746601, -0.4214933324022432, 1.5869240198367816],
], dtype=np.float64)

XYZ_TO_LMS = np.array([
    [0.8190224379967030, 0.3619062600528904, -0.1288737815209879],
    [0.0329836539323885, 0.9292868615863434, 0.0361446663506424],
    [0.0481771893596242, 0.2642395317527308, 0.6335478284694309],
], dtype=np.float64)

LMS_TO_OKLAB = np.array([
    [0.2104542683093140, 0.7936177747023054, -0.0040720430116193],
    [1.9779985324311684, -2.4285922420485799, 0.4505937096174110],
    [0.0259040424655478, 0.7827717124575296, -0.8086757549230774],
], dtype=np.float64)


def fail(message):
    raise SystemExit(message)


def read(path):
    return path.read_text(encoding="utf-8")


def parse_mat3(source, name):
    match = re.search(
        rf"const mat3 {name} = mat3\(\s*"
        r"([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*"
        r"([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*"
        r"([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\s*\);",
        source,
    )
    if match is None:
        fail(f"missing matrix {name}")
    values = [float(item) for item in match.groups()]
    return np.array(values, dtype=np.float64).reshape(3, 3)


def hook_descs(source):
    return re.findall(r"//!DESC\s+(.+)", source)


def section_after(source, desc):
    marker = f"//!DESC {desc}"
    start = source.find(marker)
    if start < 0:
        fail(f"missing pass {desc}")
    nxt = source.find("//!HOOK", start + len(marker))
    return source[start:] if nxt < 0 else source[start:nxt]


def xyz_to_oklab(xyz):
    # Shader matrices are multiplied on the left of a column vector.
    lms = XYZ_TO_LMS @ xyz
    root = np.sign(lms) * np.abs(lms) ** (1.0 / 3.0)
    return LMS_TO_OKLAB @ root


def oklab_to_xyz(lab):
    root = OKLAB_TO_LMS @ lab
    lms = root * root * root
    return LMS_TO_XYZ @ lms


def f_slope(x0, y0, x1, y1):
    den = x1 - x0
    if abs(den) < 1e-6:
        return 1.0
    return (y1 - y0) / den


def f_toe_rational(x, slope, x0, y0, x1, y1, overshoot):
    dx = x1 - x0
    dy = y1 - y0
    t = (x - x0) / dx
    normalized_slope = slope * dx / dy
    power = 1.0 / (1.0 + max(overshoot, 0.0))
    curvature = (normalized_slope - 1.0) / power
    denominator = (1.0 + curvature * (1.0 - t)) ** power
    return y0 + dy * t / denominator


def f_shoulder_rational(x, slope, x0, y0, x1, y1, overshoot):
    dx = x1 - x0
    dy = y1 - y0
    t = (x - x0) / dx
    normalized_slope = slope * dx / dy
    power = 1.0 / (1.0 + max(overshoot, 0.0))
    curvature = (normalized_slope - 1.0) / power
    denominator = (1.0 + curvature * t) ** power
    mapped = 1.0 - (1.0 - t) / denominator
    return y0 + dy * mapped


def evaluate_tone_curve(
    x, ib, iw, ob, ow,
    shadow_weight=0.4, highlight_weight=0.1, contrast_bias=0.0,
    shadow_overshoot=0.0, highlight_overshoot=1.0, reserve=True,
):
    midgray = 0.5 * ow
    shadow = midgray + shadow_weight * (ob - midgray)
    highlight = midgray + highlight_weight * (ow - midgray)
    target_slope = 2.0 ** contrast_bias
    x0, y0 = ib, ob
    x1 = shadow
    x2 = highlight
    x3, y3 = iw, ow
    y1 = midgray + target_slope * (x1 - midgray)
    if y1 < y0:
        y1 = y0
        x1 = midgray + (y1 - midgray) / target_slope
    y2 = midgray + target_slope * (x2 - midgray)
    if y2 > y3:
        y2 = y3
        x2 = midgray + (y2 - midgray) / target_slope

    if reserve:
        output_span = max(y3 - y0, 1e-6)
        if y1 <= y0 and x1 > x0 and abs(target_slope) > 1e-6:
            reserved = min(0.02 * output_span, 0.25 * output_span)
            lifted = min(y0 + reserved, y2)
            relocated = midgray + (lifted - midgray) / target_slope
            toe_slope = (lifted - y0) / max(relocated - x0, 1e-6)
            if lifted > y0 and relocated > x0 and relocated < x2 and toe_slope < target_slope:
                x1 = relocated
                y1 = lifted
        if y2 >= y3 and x3 > x2 and abs(target_slope) > 1e-6:
            reserved = min(0.04 * output_span, 0.25 * output_span)
            lowered = max(y3 - reserved, y1)
            relocated = midgray + (lowered - midgray) / target_slope
            shoulder_slope = (y3 - lowered) / max(x3 - relocated, 1e-6)
            if lowered < y3 and relocated > x1 and relocated < x3 and shoulder_slope < target_slope:
                x2 = relocated
                y2 = lowered

    slope = target_slope
    intercept = y1 - slope * x1
    if x1 <= x <= x2:
        return slope * x + intercept
    if x < x1:
        if y1 <= y0:
            return y0
        slope_toe = f_slope(x0, y0, x1, y1)
        if slope_toe >= slope:
            return slope * x + intercept
        return f_toe_rational(x, slope, x0, y0, x1, y1, shadow_overshoot)
    if x > x2:
        if y2 >= y3:
            return y3
        slope_shoulder = f_slope(x2, y2, x3, y3)
        if slope_shoulder >= slope:
            return slope * x + intercept
        return f_shoulder_rational(x, slope, x2, y2, x3, y3, highlight_overshoot)
    return x


def wolf_gain(delta_time, time_scale, residual, variance, duration=0.33):
    tau = max(duration * time_scale, 1.0 / 240.0)
    scaled = residual / 0.5
    weight = 1.0 / math.sqrt(1.0 + scaled * scaled)
    variance = min(max(variance, 1e-6), 1e6)
    u = weight * delta_time / tau
    decay = math.exp(-u)
    one_minus = u * (1.0 - 0.5 * u + u * u / 6.0) if u < 0.001 else 1.0 - decay
    one_minus_sq = one_minus * (1.0 + decay)
    one_plus_sq = 1.0 + decay * decay
    b = variance * weight
    denominator = one_plus_sq + b * one_minus_sq
    gain = one_minus * (one_minus + b * (1.0 + decay)) / denominator
    hyperbolic = one_minus_sq / one_plus_sq
    variance = (variance + hyperbolic / weight) / (1.0 + b * hyperbolic)
    variance = min(max(variance, 1e-6), 1e6)
    return min(max(gain, 0.0), 1.0), variance


def color_lut_hit(stored, current):
    if not stored["valid"]:
        return False
    keys = (
        "reference_white", "contrast_ratio", "hk",
        "output_min_j", "output_max_j", "output_reverse_max_jhk",
    )
    return all(stored[key] == current[key] for key in keys)


def chroma_attenuation(x, rate, threshold):
    if x >= 1.0:
        return 1.0
    if threshold >= 1.0:
        return 0.0
    norm = min(max((x - threshold) / (1.0 - threshold), 0.0), 1.0)
    smooth = norm * norm * (3.0 - 2.0 * norm)
    return smooth ** (1.0 + rate)


def boundary_search(lightness, hue, xyz_to_rgb):
    direction = np.array([math.cos(hue), math.sin(hue)])
    lower, upper = 0.0, 1.0
    for _ in range(16):
        candidate = 0.5 * (lower + upper)
        lab = np.array([lightness, candidate * direction[0], candidate * direction[1]])
        xyz = oklab_to_xyz(lab)
        rgb = xyz_to_rgb @ xyz
        inside = np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
        if inside:
            lower = candidate
        else:
            upper = candidate
    return lower


def test_structure():
    astra = read(ASTRA)
    bottosson = read(BOTTOSSON)
    for path, text in ((ASTRA, astra), (BOTTOSSON, bottosson)):
        if text.count("{") != text.count("}"):
            fail(f"unbalanced braces in {path.name}")

    b_desc = hook_descs(bottosson)
    if b_desc[:3] != [
        "gamut mapping (bottosson, boundary generation)",
        "gamut mapping (bottosson, boundary cache commit)",
        "gamut mapping (bottosson)",
    ]:
        fail(f"unexpected Bottosson pass order: {b_desc[:3]}")
    generation = section_after(bottosson, b_desc[0])
    if "gamut_boundary_valid != 0u" not in generation:
        fail("Bottosson boundary pass does not consult the cache flag")

    a_desc = hook_descs(astra)
    if "tone mapping (astra, lut generation)" not in a_desc:
        fail("missing Astra LUT pass")
    if "tone mapping (astra, chroma boundary)" not in a_desc:
        fail("missing Astra chroma boundary pass")
    if "tone mapping (astra, lut cache commit)" not in a_desc:
        fail("missing Astra LUT cache commit")
    lut_at = a_desc.index("tone mapping (astra, lut generation)")
    chroma_at = a_desc.index("tone mapping (astra, chroma boundary)")
    commit_at = a_desc.index("tone mapping (astra, lut cache commit)")
    pixel_at = a_desc.index("tone mapping (astra)")
    if not (lut_at < chroma_at < commit_at < pixel_at):
        fail("Astra cache passes are out of order")

    lut = section_after(astra, "tone mapping (astra, lut generation)")
    if "color_lut_cache_hit()" not in lut or "generate_curve_lut" not in lut:
        fail("LUT pass dropped the curve update or the cache check")
    hit = lut.find("if (color_lut_cache_hit())")
    curve = lut.find("generate_curve_lut(atlas_position)")
    if hit < 0 or curve < hit:
        fail("curve row is inside the color-table cache return")
    chroma = section_after(astra, "tone mapping (astra, chroma boundary)")
    chroma_hook = chroma[chroma.find("void hook()"):]
    if chroma_hook.find("find_chroma_boundary(") < chroma_hook.find("color_lut_cache_hit()"):
        fail("chroma boundary search is not behind the cache check")

    print("structure: passes, cache order, and braces")


def test_matrix_and_roundtrip():
    source = read(BOTTOSSON)
    pasted = parse_mat3(source, "BT2020_TO_BT709")
    product = XYZ_TO_BT709 @ BT2020_TO_XYZ
    if np.max(np.abs(pasted - product)) > 1e-12:
        fail(f"BT2020_TO_BT709 disagrees with the matrix product:\n{pasted - product}")

    rng = np.random.default_rng(15)
    rgb709 = rng.random((4096, 3))
    xyz = np.linalg.inv(XYZ_TO_BT709) @ rgb709.T
    rgb2020 = np.linalg.inv(BT2020_TO_XYZ) @ xyz
    recovered = (pasted @ rgb2020).T
    rgb709_error_src = rgb709
    matrix_error = np.max(np.abs(recovered - rgb709_error_src))
    if matrix_error > 1e-9:
        fail(f"matrix round trip error {matrix_error}")

    lab = np.array([xyz_to_oklab(column) for column in xyz.T])
    xyz_back = np.array([oklab_to_xyz(row) for row in lab]).T
    round_trip = (XYZ_TO_BT709 @ xyz_back).T
    round_error = np.max(np.abs(round_trip - rgb709))
    if not round_error > matrix_error:
        fail("Oklab round trip is not the larger error")
    if round_error > 2e-3:
        fail(f"Oklab round trip error {round_error} is larger than expected")
    print(
        f"color: matrix error {matrix_error:.3e}, "
        f"Oklab round-trip error {round_error:.3e}"
    )


def test_gamut_boundary():
    samples = []
    for lightness in (0.05, 0.2, 0.45, 0.7, 0.9):
        for hue in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            wide = boundary_search(lightness, hue, np.linalg.inv(BT2020_TO_XYZ))
            narrow = boundary_search(lightness, hue, XYZ_TO_BT709)
            if not np.isfinite(wide) or not np.isfinite(narrow):
                fail("non-finite gamut boundary")
            if narrow > wide + 1e-4:
                fail(f"BT.709 boundary {narrow} exceeds BT.2020 {wide}")
            samples.append((lightness, hue, wide, narrow))
    print(f"gamut: {len(samples)} hue/lightness cells, BT.2020 contains BT.709")


def test_cache_key():
    current = {
        "valid": True,
        "reference_white": 203.0,
        "contrast_ratio": 1000.0,
        "hk": 1.0,
        "output_min_j": 0.01,
        "output_max_j": 0.42,
        "output_reverse_max_jhk": 0.43,
    }
    if color_lut_hit({"valid": False, **{k: current[k] for k in current if k != "valid"}}, current):
        fail("an empty cache must miss")
    stored = dict(current)
    if not color_lut_hit(stored, current):
        fail("matching display settings must hit")
    changed = dict(current)
    changed["reference_white"] = 400.0
    if color_lut_hit(stored, changed):
        fail("a white-point change must miss")
    nan_key = dict(current)
    nan_key["output_min_j"] = math.nan
    if color_lut_hit(stored, nan_key):
        fail("a non-finite output range must miss")
    print("cache: miss, hit, white-point change, non-finite range")


def test_tone_curve():
    scenes = []
    for ib, iw, ob, ow in (
        (0.0, 0.55, 0.0, 0.40),
        (0.02, 0.80, 0.0, 0.50),
        (0.0, 0.30, 0.01, 0.45),
        (0.05, 0.90, 0.0, 0.35),
    ):
        xs = np.linspace(ib, iw, 513)
        old = np.array([evaluate_tone_curve(x, ib, iw, ob, ow, reserve=False) for x in xs])
        new = np.array([evaluate_tone_curve(x, ib, iw, ob, ow, reserve=True) for x in xs])
        if np.max(np.abs(old - new)) != 0.0:
            fail(f"default curve changed for scene {(ib, iw, ob, ow)}")
        if np.any(np.diff(new) < -1e-12):
            fail("default curve decreased")
        if ob == 0.0 and abs(evaluate_tone_curve(0.0, ib, iw, ob, ow)) > 1e-9:
            fail("true black moved on a default scene")
        scenes.append((ib, iw, ob, ow))

    ib, iw, ob, ow = 0.0, 0.80, 0.0, 0.50
    xs = np.linspace(0.50, iw, 257)
    old = np.array([
        evaluate_tone_curve(
            x, ib, iw, ob, ow, highlight_weight=1.0, reserve=False
        ) for x in xs
    ])
    new = np.array([
        evaluate_tone_curve(
            x, ib, iw, ob, ow, highlight_weight=1.0, reserve=True
        ) for x in xs
    ])
    if np.max(new) - np.min(new) <= 1e-4:
        fail("collapsed shoulder is still flat")
    if np.any(np.diff(new) < -1e-12):
        fail("highlight reserve decreased")
    if abs(new[-1] - ow) > 1e-6:
        fail("highlight reserve missed output white")
    if np.unique(np.round(old, 8)).size > 3 and np.max(old) - np.min(old) > np.max(new) - np.min(new):
        fail("old shoulder was already more separated than the reserve")

    xs = np.linspace(0.0, 0.08, 129)
    dark = np.array([
        evaluate_tone_curve(
            x, 0.0, 0.60, 0.0, 0.50, shadow_weight=1.0, reserve=True
        ) for x in xs
    ])
    if abs(dark[0]) > 1e-9:
        fail("true black moved")
    if np.any(np.diff(dark) < -1e-12):
        fail("shadow reserve decreased")
    if dark[-1] <= dark[0]:
        fail("shadow reserve did not open the toe")
    print(f"curve: {len(scenes)} default scenes unchanged, highlight and shadow reserves separate")


def test_exposure():
    variance = 1.0
    ev = 0.0
    target = 1.5
    duration = 0.33
    previous = ev
    for frame in range(1, 48):
        gain, variance = wolf_gain(1.0 / 60.0, 0.50, target - ev, variance, duration)
        ev = ev + (target - ev) * gain
        if gain < 0.0 or gain > 1.0:
            fail("exposure gain left [0, 1]")
        if (ev - previous) < -1e-12:
            fail("exposure reversed while rising")
        if (ev - target) * (previous - target) < -1e-9:
            fail("exposure crossed the target")
        previous = ev
    if abs(ev - target) > 0.2:
        fail(f"exposure did not approach the target, ended at {ev}")

    held, _ = wolf_gain(0.0, 0.50, 0.4, variance, duration)
    if held > 1e-6:
        fail("a zero time step still moved exposure")

    # A gap longer than the time constant is the shader's cut reset.
    if not (0.5 > duration):
        fail("the cut example is not past the time constant")
    print(f"exposure: monotone approach to {target}, hold on a repeated timestamp")


def test_chroma_and_fallback():
    xs = np.linspace(0.0, 1.2, 25)
    values = [chroma_attenuation(x, 1.0, 0.0) for x in xs]
    if any(not 0.0 <= value <= 1.0 for value in values):
        fail("chroma attenuation left [0, 1]")
    if chroma_attenuation(1.0, 3.0, 0.2) != 1.0:
        fail("identity endpoint changed")
    if chroma_attenuation(0.2, 1.0, 1.0) != 0.0:
        fail("hard threshold changed")
    if any(b < a - 1e-12 for a, b in zip(values, values[1:])):
        fail("chroma attenuation decreased as lightness ratio rose")

    def search(valid):
        lower, upper = 0.0, 1.0
        chosen = 0.0
        steps = 0
        for _ in range(12):
            steps += 1
            middle = 0.5 * (lower + upper)
            if valid(middle):
                lower = middle
                chosen = middle
            else:
                upper = middle
        return chosen, steps

    chosen, steps = search(lambda scale: scale <= 0.3)
    if steps != 12:
        fail("fallback search length changed")
    if abs(chosen - 0.3) > 2.0 ** -11:
        fail(f"fallback search missed the valid edge: {chosen}")
    print("chroma: attenuation stays ordered, fallback search keeps 12 steps")


def test_static_volume():
    astra = read(ASTRA)
    match = re.search(
        r"//!SAVE LUTS\s+"
        r"//!WIDTH\s+(\d+)\s+"
        r"//!HEIGHT\s+(\d+)\s+"
        r"//!COMPUTE\s+\d+\s+\d+\s+"
        r"//!DESC tone mapping \(astra, lut generation\)",
        astra,
    )
    if match is None:
        fail("could not read the LUT atlas size")
    width, height = int(match.group(1)), int(match.group(2))
    color_texels = width * (height - 1)
    curve_texels = 1024
    if color_texels != 819650 or curve_texels != 1024:
        fail(f"unexpected atlas split {color_texels} / {curve_texels}")
    print(
        f"volume: {color_texels} cached color texels, "
        f"{curve_texels} live curve texels, "
        "46569 chroma cells, 32768 Bottosson cells"
    )


def main():
    test_structure()
    test_matrix_and_roundtrip()
    test_gamut_boundary()
    test_cache_key()
    test_tone_curve()
    test_exposure()
    test_chroma_and_fallback()
    test_static_volume()
    print("all plan checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
