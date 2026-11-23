import unittest

from advisor.ingestion import chunk_text, extract_text


class IngestionTests(unittest.TestCase):
    def test_extracts_text_and_csv_rows(self) -> None:
        self.assertEqual(extract_text("notes.txt", b"  sprint update  "), "sprint update")
        self.assertEqual(extract_text("notes.md", b"# Sprint update"), "# Sprint update")
        self.assertEqual(
            extract_text("risks.csv", b"risk,owner\nLate API,Sam\n"),
            "risk | owner\nLate API | Sam",
        )

    def test_chunks_with_overlap(self) -> None:
        chunks = chunk_text("one two three four five six", chunk_size=15, overlap=5)
        self.assertEqual(chunks, ["one two three", "four five six"])

    def test_rejects_invalid_chunk_configuration(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("some text", chunk_size=4, overlap=4)

    def test_rejects_unsupported_file_types(self) -> None:
        with self.assertRaises(ValueError):
            extract_text("archive.zip", b"content")


if __name__ == "__main__":
    unittest.main()