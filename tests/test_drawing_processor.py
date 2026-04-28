from app.services.drawing_processor import build_drawing_chunks, extract_entities


def test_extract_entities_preserves_architectural_ids():
    text = "See sheet AR-402/7 for W126 near door D101. Keynote 3.23 appears on A-051."

    entities = extract_entities(text, sheet_number="AR-601")

    assert "AR-601" in entities
    assert "AR-402/7" in entities
    assert "W126" in entities
    assert "D101" in entities
    assert "3.23" in entities
    assert "A-051" in entities


def test_build_drawing_chunks_adds_sheet_page_and_source_metadata():
    chunks = build_drawing_chunks(
        filename="cedar-ridge.pdf",
        project_id=10,
        document_summary={
            "overall_project_scope": "Exterior rehabilitation.",
            "important_work_packages": ["Window well replacement"],
            "sheets": [{"sheet_number": "AR-601", "title": "Window Schedule", "page": 9}],
            "all_keynotes": ["3.23: Install new metal window well"],
            "window_door_schedule_highlights": ["W126 is aluminum clad"],
            "construction_constraints": ["Coordinate excavation around existing foundation"],
            "normalized_entities": {
                "sheets": ["AR-601"],
                "keynotes": ["3.23"],
                "windows": ["W126"],
                "doors": [],
                "details": ["AR-402/7"],
            },
        },
        page_records=[
            {
                "page": 9,
                "raw_text": "AR-601 W126 3.23 AR-402/7",
                "image_path": "uploads/1/10/document_1_pages/page_0009.png",
                "entities": ["AR-601", "W126", "3.23", "AR-402/7"],
                "analysis": {
                    "sheet_number": "AR-601",
                    "title": "Window Schedule",
                    "summary": "Window and keynote schedule.",
                    "scope_items": ["Replace window well at W126"],
                    "keynotes": ["3.23: Install new metal window well"],
                    "visible_callouts": ["AR-402/7"],
                    "tables": [{"title": "Window Schedule", "rows": [{"mark": "W126", "type": "fixed"}]}],
                    "materials": ["Metal window well"],
                    "locations": ["North elevation"],
                    "risks_or_constraints": ["Protect existing foundation"],
                },
            }
        ],
    )

    chunk_types = {chunk.metadata["chunk_type"] for chunk in chunks}
    assert "document_summary" in chunk_types
    assert "sheet_summary" in chunk_types
    assert "keynote" in chunk_types
    assert "schedule_row" in chunk_types
    assert "construction_detail" in chunk_types

    schedule_chunk = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "schedule_row")
    assert schedule_chunk.metadata["sheet"] == "AR-601"
    assert schedule_chunk.metadata["page"] == 9
    assert schedule_chunk.metadata["source_image"].endswith("page_0009.png")
    assert "W126" in schedule_chunk.metadata["entities"]
