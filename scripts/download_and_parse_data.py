import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("download_and_parse_data")

DATA_DIRECTORY = Path("data")
SPLITS = ["train", "dev", "test"]
EXTENSION = ".txt"

Word = str
Label = str
Row = tuple[Word, Label]
Sentence = list[Row]


def get_remote_files_urls() -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    BASE_PATH = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{placeholder}.conllu"
    return (
        ("train", BASE_PATH.format(placeholder="train")),
        ("dev", BASE_PATH.format(placeholder="dev")),
        ("test", BASE_PATH.format(placeholder="test")),
    )


def parse_conllu(full_text: str) -> list[Sentence]:
    lines = full_text.splitlines()
    parsed_sentences = []
    sentence = []
    for line in lines:
        # CoNLL-U format gives an index to each word, so we care only about lines with a number as the first character
        if line and line[0].isnumeric():
            # All interesting values are separated by a tab character, so we split on that
            cells = line.split("\t")
            # We are interested only in the word and the Universal POS tag, which are respectively at columns 1 and 3
            word, pos_label = cells[1], cells[3]
            sentence.append((word, pos_label))
        else:
            # We need to add the sentence to the list of parsed sentences, only if the sentence is actually something
            if len(sentence) > 0:
                parsed_sentences.append(sentence)
                sentence = []

    # Add any sentence that remains in the accumulator
    if sentence:
        parsed_sentences.append(sentence)

    return parsed_sentences


def save_sentences_to_file(sentences: list[Sentence], output_path: Path):
    with open(output_path, "w+") as f:
        for sentence in sentences:
            for word, tag in sentence:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")


def main():
    # Define file paths
    file_paths = [DATA_DIRECTORY / f"{split}{EXTENSION}" for split in SPLITS]

    # Check if data directory and splits have already been retrieved
    if all(file_path.exists() for file_path in file_paths):
        console_logger.info("All datafiles already retrieved, done ✅")
        return

    # There are some files missing, redownload all of them
    DATA_DIRECTORY.mkdir(exist_ok=True)
    remote_file_urls = get_remote_files_urls()
    for split, url in remote_file_urls:
        console_logger.info(f"Downloading split {split}...")
        response = requests.get(url)

        # Parse sentences from downloaded file
        console_logger.info("Parsing split...")
        split_sentences = parse_conllu(response.text)

        # Save to given path
        output_path = DATA_DIRECTORY / f"{split}{EXTENSION}"
        console_logger.info(f"Saving split {split} to path {str(output_path)}...")
        save_sentences_to_file(split_sentences, output_path)

        console_logger.info(f"Finished downloading split {split}.")

    console_logger.info("Finished all splits ✅")


if __name__ == "__main__":
    main()
