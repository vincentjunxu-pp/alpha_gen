from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "results" / "typed_tree_gp"
CSV = BASE / "typed_tree_gp_clean_run_final_population.csv"
OUT = BASE / "analysis"
DEDUPE = True
FONT = "C:/Windows/Fonts/arial.ttf"


def font(size, bold=False):
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else FONT
    return ImageFont.truetype(path, size)


def text_rot(img, xy, text, fnt, fill=(45, 45, 45), angle=35):
    box = Image.new("RGBA", (360, 80), (255, 255, 255, 0))
    ImageDraw.Draw(box).text((0, 0), text, font=fnt, fill=fill)
    img.alpha_composite(box.rotate(angle, expand=True), xy)


def walk(node, fields, unary, combo):
    if node["node"] == "field":
        fields[node["field"]] += 1
    elif node["node"] == "unary":
        unary[node["op"]] += 1
        walk(node["child"], fields, unary, combo)
    elif node["node"] == "binary":
        combo[f"binary:{node['op']}"] += 1
        walk(node["left"], fields, unary, combo)
        walk(node["right"], fields, unary, combo)


def bar(counter, path, title, top=None):
    items = Counter(counter).most_common(top)
    if not items:
        items = [("none", 0)]
    w, row, left, right, topy = 1700, 44, 480, 100, 115
    h = topy + row * len(items) + 70
    img = Image.new("RGBA", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((35, 28), title, font=font(32, True), fill=(30, 34, 40))
    mx = max(v for _, v in items) or 1
    for i, (k, v) in enumerate(items):
        y = topy + i * row
        bw = int((w - left - right) * v / mx)
        d.text((35, y + 5), str(k), font=font(18), fill=(45, 45, 45))
        d.rounded_rectangle((left, y, left + bw, y + 27), radius=5, fill=(76, 120, 168, 215))
        d.text((left + bw + 10, y + 3), str(v), font=font(17), fill=(45, 45, 45))
    img.convert("RGB").save(path, quality=95)


def stats(vals):
    s = pd.Series(vals).dropna()
    if s.empty:
        return None
    return s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.max()


def boxplot(df, group, value, hue, colors, path, title, ylabel):
    groups, hues = list(dict.fromkeys(df[group])), list(colors)
    vals = df[value].dropna()
    y0, y1 = 0, float(vals.max() * 1.12 if len(vals) else 1)
    w, h, L, R, T, B = 1800, 940, 110, 50, 95, 230
    img = Image.new("RGBA", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((35, 28), title, font=font(32, True), fill=(30, 34, 40))
    d.text((35, 70), ylabel, font=font(18), fill=(80, 80, 80))
    def yy(x): return int(h - B - (x - y0) / (y1 - y0) * (h - T - B))
    for t in [i * y1 / 5 for i in range(6)]:
        y = yy(t)
        d.line((L, y, w - R, y), fill=(224, 228, 235), width=1)
        d.text((30, y - 10), f"{t:.3f}", font=font(15), fill=(90, 90, 90))
    step = (w - L - R) / max(len(groups), 1)
    bw = min(58, step / (len(hues) + 1))
    for gi, g in enumerate(groups):
        xc = L + step * (gi + .5)
        text_rot(img, (int(xc - 38), h - B + 35), g, font(16), angle=35)
        for hi, hu in enumerate(hues):
            st = stats(df[(df[group] == g) & (df[hue] == hu)][value])
            if not st:
                continue
            mn, q1, med, q3, mx = st
            x = int(xc + (hi - (len(hues) - 1) / 2) * bw * 1.25)
            c = colors[hu]
            d.line((x, yy(mn), x, yy(mx)), fill=c, width=2)
            d.rectangle((x - bw // 2, yy(q3), x + bw // 2, yy(q1)), fill=(*c, 90), outline=c, width=2)
            d.line((x - bw // 2, yy(med), x + bw // 2, yy(med)), fill=c, width=3)
            d.line((x - bw // 3, yy(mn), x + bw // 3, yy(mn)), fill=c, width=2)
            d.line((x - bw // 3, yy(mx), x + bw // 3, yy(mx)), fill=c, width=2)
    for i, hu in enumerate(hues):
        x = w - 330 + i * 150
        d.rounded_rectangle((x, 48, x + 28, 68), radius=4, fill=(*colors[hu], 110), outline=colors[hu])
        d.text((x + 38, 45), hu, font=font(18), fill=(45, 45, 45))
    img.convert("RGB").save(path, quality=95)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    if DEDUPE:
        df = df.drop_duplicates("expression").copy()

    long = df.melt("tree_mode", ["train_abs_rank_ic", "valid_abs_rank_ic"], "sample", "abs_rank_ic")
    long["sample"] = long["sample"].str.replace("_abs_rank_ic", "", regex=False)
    boxplot(long, "tree_mode", "abs_rank_ic", "sample",
            {"train": (76, 120, 168), "valid": (245, 133, 24)},
            OUT / "box_abs_rank_ic_by_mode.png", "Train vs Valid Abs RankIC by Tree Mode", "Abs RankIC")

    fields, unary, combo = Counter(), Counter(), Counter()
    for _, r in df.iterrows():
        combo[f"root:{r['tree_combiner']}"] += 1
        for n in json.loads(r["tree_slots"]).values():
            walk(n, fields, unary, combo)
    bar(fields, OUT / "bar_top20_fields.png", "Top 20 Field Usage", 20)
    bar(unary, OUT / "bar_unary_ops.png", "Unary Operator Usage")
    bar(combo, OUT / "bar_combo_ops.png", "Combination Operator Usage")

    barra = Counter(x.strip() for s in df["train_barra_selected_styles"].fillna("")
                    for x in str(s).split(",") if x.strip())
    bar(barra, OUT / "bar_barra_styles.png", "Selected Barra Style Usage")

    neu = df.melt("tree_mode", ["train_abs_rank_ic", "train_neutralized_abs_rank_ic"], "metric", "abs_rank_ic")
    neu["metric"] = neu["metric"].map({"train_abs_rank_ic": "raw", "train_neutralized_abs_rank_ic": "neutralized"})
    boxplot(neu, "tree_mode", "abs_rank_ic", "metric",
            {"raw": (84, 162, 75), "neutralized": (228, 87, 86)},
            OUT / "box_raw_vs_neutralized_train_abs_ic.png", "Raw vs Neutralized Train Abs RankIC by Tree Mode", "Train Abs RankIC")
    print(f"saved plots to {OUT.resolve()}")


if __name__ == "__main__":
    main()
