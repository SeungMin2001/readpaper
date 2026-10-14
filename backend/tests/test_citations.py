import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from readpaper.citations import parse_grobid_references, title_similarity  # noqa: E402
from readpaper.citations import parse_grobid_paper  # noqa: E402
from collect_cited_fulltext import _crossref_pdf_locations, _select_crossref_match  # noqa: E402


class CitationParsingTests(unittest.TestCase):
    def test_parses_bibliography_title_authors_year_and_arxiv_version(self):
        tei = '''<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0"
             xmlns:xml="http://www.w3.org/XML/1998/namespace">
          <text><back><listBibl>
            <biblStruct xml:id="b0">
              <analytic><title>Convolutional sequence to sequence learning</title>
                <author><persName><forename>Jonas</forename><surname>Gehring</surname></persName></author>
              </analytic>
              <monogr><imprint><date when="2017">2017</date></imprint></monogr>
              <note type="raw_reference">Jonas Gehring. Convolutional sequence to sequence learning. arXiv:1705.03122v2, 2017.</note>
            </biblStruct>
          </listBibl></back></text>
        </TEI>'''

        [reference] = parse_grobid_references(tei)
        self.assertEqual(reference["number"], 1)
        self.assertEqual(reference["title"], "Convolutional sequence to sequence learning")
        self.assertEqual(reference["authors"], ["Jonas Gehring"])
        self.assertEqual(reference["year"], 2017)
        self.assertEqual(reference["arxiv_id"], "1705.03122")
        self.assertEqual(reference["arxiv_version"], 2)
        self.assertIn("raw_reference", reference)

    def test_parses_root_metadata_for_a_non_attention_paper(self):
        tei = '''<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc>
            <titleStmt><title>Provided proper attribution is provided, the long usage notice grants permission to reproduce figures for scholarly works. A General Example Paper</title></titleStmt>
            <sourceDesc><biblStruct><analytic>
              <author><persName><forename>Alex</forename><surname>Reader</surname></persName></author>
            </analytic></biblStruct></sourceDesc>
            <publicationStmt><idno type="DOI">10.1234/example.1</idno>
              <idno type="arXiv">arXiv:2401.12345v2</idno></publicationStmt>
          </fileDesc><profileDesc><abstract><p>A generic abstract.</p></abstract></profileDesc></teiHeader>
          <text><back><listBibl><biblStruct xml:id="b0">
            <analytic><title>Some cited paper</title></analytic>
            <note type="raw_reference">A. Author. Some cited paper. 2020.</note>
          </biblStruct></listBibl></back></text>
        </TEI>'''

        paper = parse_grobid_paper(tei)
        self.assertEqual(paper["title"], "A General Example Paper")
        self.assertEqual(paper["authors"], ["Alex Reader"])
        self.assertEqual(paper["doi"], "10.1234/example.1")
        self.assertEqual(paper["arxiv_id"], "2401.12345")
        self.assertEqual(paper["abstract"], "A generic abstract.")
        self.assertEqual(paper["references"][0]["title"], "Some cited paper")

    def test_title_score_handles_a_short_cited_title_without_accepting_wrong_work(self):
        self.assertGreaterEqual(
            title_similarity("A decomposable attention model", "A Decomposable Attention Model for Natural Language Inference"),
            0.65,
        )
        self.assertLess(
            title_similarity("Structured attention networks", "Learning Structured Text Representations"),
            0.6,
        )

    def test_crossref_fallback_matches_title_author_and_year(self):
        reference = {
            "title": "A decomposable attention model",
            "authors": ["Ankur Parikh"],
            "year": 2016,
        }
        works = [
            {
                "DOI": "10.0000/wrong",
                "title": ["A Decomposable Attention Model for Natural Language Inference"],
                "author": [{"given": "Someone", "family": "Else"}],
                "published": {"date-parts": [[2016]]},
            },
            {
                "DOI": "10.18653/v1/d16-1244",
                "title": ["A Decomposable Attention Model for Natural Language Inference"],
                "author": [{"given": "Ankur", "family": "Parikh"}],
                "published": {"date-parts": [[2016]]},
                "link": [{
                    "URL": "https://aclanthology.org/D16-1244.pdf",
                    "content-type": "application/pdf",
                }],
            },
        ]
        match = _select_crossref_match(reference, works)
        self.assertEqual(match["DOI"], "10.18653/v1/d16-1244")
        self.assertEqual(_crossref_pdf_locations(match)[0]["pdf_url"], "https://aclanthology.org/D16-1244.pdf")


if __name__ == "__main__":
    unittest.main()
