import argparse
import pandas as pd
import random
from pathlib import Path


# Template sentences for synthetic review generation
TEMPLATES = {
    "happy": [
        "I really enjoyed this, it made my day!",
        "This put a smile on my face. Loved it!",
        "A very pleasant experience, felt great afterwards.",
        "Awesome feeling! Totally worth it."
    ],
    "sad": [
        "I expected better, this made me feel down.",
        "This did not go well, quite disappointing.",
        "Kind of depressing, I didn’t like it.",
        "It left me sad, could have been better."
    ],
    "angry": [
        "This was frustrating and annoying.",
        "I’m really upset with how this turned out.",
        "This experience made me so angry!",
        "I wasted my time, terrible outcome."
    ],
    "neutral": [
        "It was okay, nothing special really.",
        "Not too bad, not too good, kind of average.",
        "I don’t feel strongly about this either way.",
        "Very neutral experience overall."
    ],
    "fear": [
        "This made me anxious, not a pleasant moment.",
        "I felt uneasy and scared during the experience.",
        "It gave me chills, kind of terrifying.",
        "I wouldn’t want to go through that again."
    ],
    "disgust": [
        "Really unpleasant, I disliked it a lot.",
        "It felt gross and uncomfortable to me.",
        "A very disgusting experience, not for me.",
        "I honestly felt sickened by it."
    ],
    "surprise": [
        "Wow! Totally unexpected in a good way!",
        "That caught me off guard, nice surprise!",
        "I didn’t see that coming, but I liked it!",
        "A surprising twist that worked out well!"
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic text reviews using predicted emotions.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the predictions CSV file.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV path (relative or absolute). Defaults to outputs/reviews_<model>_test.csv",
    )
    return parser.parse_args()


def generate_review(emotion: str) -> str:
    """
    Pick a random review sentence based on the emotion class.
    """
    emotion = emotion.lower()
    if emotion not in TEMPLATES:
        return "No review available."
    return random.choice(TEMPLATES[emotion])


def main():
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"Loaded predictions: {input_csv}, rows={len(df)}")

    if "predicted_label" not in df.columns:
        raise KeyError("CSV must contain 'predicted_label' column.")

    # Generate reviews
    df["review_text"] = df["predicted_label"].apply(generate_review)

    # Determine automatic output name if not provided
    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
    else:
        output_csv = input_csv.parent / f"reviews_{input_csv.stem.replace('predictions_', '')}.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Saved reviews to: {output_csv}")


if __name__ == "__main__":
    main()
