from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image


@dataclass
class OcrWord:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float
    block: int | None = None
    paragraph: int | None = None
    line: int | None = None


@dataclass
class OcrLine:
    page: int
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    ocr_confidence: float
    word_count: int
    source_image: str
    source_variant: str
    psm: int


class OcrTextExtractor:
    def extract_words(
        self,
        image_path: str,
        lang: str = "tur+eng",
        psm: int = 6,
    ) -> list[OcrWord]:
        image = Image.open(image_path)

        config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"

        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[OcrWord] = []
        total = len(data["text"])

        for index in range(total):
            text = str(data["text"][index]).strip()

            if not text:
                continue

            try:
                confidence = float(data["conf"][index])
            except Exception:
                confidence = -1.0

            if confidence < 0:
                continue

            x = int(data["left"][index])
            y = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])

            words.append(
                OcrWord(
                    text=text,
                    x0=x,
                    y0=y,
                    x1=x + width,
                    y1=y + height,
                    confidence=confidence / 100,
                    block=data.get("block_num", [None] * total)[index],
                    paragraph=data.get("par_num", [None] * total)[index],
                    line=data.get("line_num", [None] * total)[index],
                )
            )

        return words

    def words_to_lines(
        self,
        words: list[OcrWord],
        page: int,
        image_path: str,
        variant: str,
        psm: int,
    ) -> list[OcrLine]:
        if not words:
            return []

        sorted_words = sorted(words, key=lambda word: (word.y0, word.x0))

        heights = [max(1, word.y1 - word.y0) for word in sorted_words]
        median_height = sorted(heights)[len(heights) // 2]
        y_tolerance = max(6, median_height * 0.55)

        groups: list[dict] = []

        for word in sorted_words:
            center_y = (word.y0 + word.y1) / 2

            best_group = None
            best_distance = None

            for group in groups:
                distance = abs(group["center_y"] - center_y)

                if distance <= y_tolerance and (
                    best_distance is None or distance < best_distance
                ):
                    best_group = group
                    best_distance = distance

            if best_group is None:
                groups.append(
                    {
                        "center_y": center_y,
                        "words": [word],
                    }
                )
            else:
                best_group["words"].append(word)
                centers = [
                    (item.y0 + item.y1) / 2
                    for item in best_group["words"]
                ]
                best_group["center_y"] = sum(centers) / len(centers)

        lines: list[OcrLine] = []

        for group in sorted(groups, key=lambda item: item["center_y"]):
            line_words = sorted(group["words"], key=lambda word: word.x0)

            text = " ".join(word.text for word in line_words)
            text = " ".join(text.split()).strip()

            if not text:
                continue

            avg_confidence = (
                sum(word.confidence for word in line_words) / len(line_words)
                if line_words
                else 0.0
            )

            lines.append(
                OcrLine(
                    page=page,
                    text=text,
                    x0=min(word.x0 for word in line_words),
                    y0=min(word.y0 for word in line_words),
                    x1=max(word.x1 for word in line_words),
                    y1=max(word.y1 for word in line_words),
                    ocr_confidence=round(avg_confidence, 4),
                    word_count=len(line_words),
                    source_image=Path(image_path).name,
                    source_variant=variant,
                    psm=psm,
                )
            )

        return lines

    def extract_lines(
        self,
        image_path: str,
        page: int,
        variant: str,
        lang: str = "tur+eng",
        psm: int = 6,
    ) -> list[OcrLine]:
        words = self.extract_words(
            image_path=image_path,
            lang=lang,
            psm=psm,
        )

        return self.words_to_lines(
            words=words,
            page=page,
            image_path=image_path,
            variant=variant,
            psm=psm,
        )