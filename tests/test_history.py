from database.history import HistoryDB


def test_session_lifecycle_and_stats(db: HistoryDB):
    db.upsert_document({"document_id": "d1", "document_name": "a.pdf", "title": "A", "authors": ["X"],
                        "year": "2020", "source_type": "upload", "page_count": 3, "char_count": 900,
                        "file_hash": "h"})
    db.create_session("s1", "question one", "APA", 12)
    db.link_session_document("s1", "d1")
    db.record_event("s1", "plan", "completed", "3 subtasks")
    db.finish_session("s1", "completed", registry=[{"source_id": "S1"}], quality={"score": 90},
                      report_markdown="# R\n", num_chunks=7)

    db.create_session("s2", "question two", "IEEE", 12)
    db.finish_session("s2", "failed", error="LLM timed out")

    session = db.get_session("s1")
    assert session["status"] == "completed"
    assert session["registry"] == [{"source_id": "S1"}]
    assert session["quality"]["score"] == 90
    assert session["documents"][0]["document_id"] == "d1"
    assert db.session_events("s1")[0]["agent"] == "plan"

    listed = {s["session_id"]: s for s in db.list_sessions()}
    assert listed["s1"]["quality_score"] == 90 and listed["s1"]["has_report"] is True
    assert listed["s2"]["error"] == "LLM timed out"

    stats = db.stats()
    assert stats == {"total_sessions": 2, "completed_sessions": 1, "reports_generated": 1, "papers_processed": 1}
    assert db.list_documents()[0]["session_count"] == 1


def test_interrupted_sessions_are_marked_on_restart(db: HistoryDB):
    db.create_session("s3", "crashed run", "APA", 8)
    assert db.mark_interrupted_sessions() == 1
    assert db.get_session("s3")["status"] == "interrupted"
