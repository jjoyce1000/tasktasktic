#!/usr/bin/env python3
"""
PDF to CSV Converter for TaskTastic Import

Parses PDF files (syllabi, schedules, calendars) and outputs CSV in the format
expected by the import capability:

  Category,Tag,Task,Due Date,Status
  School,<course name>,<task>,<due date>,Open

- Category: Always "School"
- Tag: Course name extracted from the PDF
- Task: Specific task or entry for a particular date
- Due Date: Date in M/D/YYYY format
- Status: Always "Open"

When ANTHROPIC_API_KEY is set, uses Claude Sonnet 4.6 for advanced interpretation of complex
layouts, prose, and varied syllabus formats. Falls back to rule-based parsing otherwise.
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Model for PDF interpretation (Claude Sonnet 4.6 for best accuracy on complex documents)
# Anthropic API uses "claude-sonnet-4-6"; "anthropic.claude-sonnet-4-6" is AWS Bedrock format
def _get_pdf_ai_model() -> str:
    raw = (os.environ.get("PDF_AI_MODEL", "claude-sonnet-4-6") or "").strip()
    return raw[10:] if raw.startswith("anthropic.") else raw


def _has_anthropic_key() -> bool:
    """Check if Anthropic API key is set (supports ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN)."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
    return len(key) > 0


def _parse_with_ai(text: str, tables_str: str, filename: str) -> list | None:
    """
    Use Claude Sonnet 4.6 to interpret PDF content and extract tasks.
    Returns list of {task, date, course} or None on failure.
    """
    if not anthropic or not _has_anthropic_key():
        return None

    # Truncate if too long (Claude context ~200k, keep under 100k chars for safety)
    content = f"Filename: {filename}\n\n--- Extracted Text ---\n{text}"
    if tables_str:
        content += f"\n\n--- Extracted Tables ---\n{tables_str}"
    if len(content) > 90000:
        content = content[:90000] + "\n\n[... content truncated ...]"

    prompt = """Extract all tasks, assignments, due dates, exams, readings, and schedule entries from this syllabus/schedule PDF.
Return a JSON object with a "tasks" array. Each task must have:
- "task": short description (e.g. "Homework 1", "Exam 2", "Read Ch. 5")
- "date": YYYY-MM-DD format, or empty string if no date
- "course": course name/code (e.g. M156, Physics 111) or "General"

Rules:
- Infer year from document (Fall 2026, Spring 2026, etc.) when dates lack year
- One item per task; split "HW 1, 2, 3" into separate tasks if dates differ
- Skip headers, column labels, and meta text (e.g. "Week", "Monday", "Date")
- Use the course name from the document title or header
- Dates: prefer ISO YYYY-MM-DD; empty string if unknown
- Include ALL weeks from the start: calendar grids often have Mon–Fri columns; the first date in each row is Monday (e.g. 1/12). Do NOT skip Week 1 or the first week's content.
- CRITICAL – Date alignment in schedule grids: When the document has a grid with date columns (e.g. Mon 1/12 | Tue 1/13 | Wed 1/14 | Thu 1/15 | Fri 1/16), each task belongs to the date of its COLUMN. The first content cell maps to Monday's date, the second to Tuesday's, the third to Wednesday's, etc. "Section 5.5 - Substitution" in the Friday column must get Friday's date (e.g. 1/16), NOT Monday's (1/12). Match each task to the date of the column it appears under.

Example output: {"tasks": [{"task": "Homework 1 due", "date": "2026-01-15", "course": "M156"}, ...]}"""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_get_pdf_ai_model(),
            max_tokens=4096,
            system="You extract structured task data from syllabi and schedules. Respond only with valid JSON.",
            messages=[{"role": "user", "content": f"{prompt}\n\n{content}"}],
            temperature=0.1,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            inp = getattr(usage, "input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
            print(f"[PDF AI] tokens: input={inp} output={out} total={inp + out}", file=sys.stderr)
        text_block = next((b for b in resp.content if getattr(b, "type", None) == "text"), None)
        raw = text_block.text if text_block else None
        if not raw:
            return None
        # Strip markdown code blocks (```json ... ``` or ``` ... ```)
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*\r?\n?", "", raw, flags=re.I)
        raw = re.sub(r"\r?\n?```\s*$", "", raw)
        raw = raw.strip()
        # Fallback: if still has backticks or doesn't start with {, extract JSON object
        if "`" in raw or not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start : end + 1]
        data = json.loads(raw)
        tasks = data.get("tasks") or []
        if not isinstance(tasks, list):
            return None
        items = []
        for t in tasks:
            if isinstance(t, dict) and t.get("task"):
                items.append({
                    "task": str(t["task"]).strip(),
                    "date": (t.get("date") or "").strip(),
                    "course": (t.get("course") or "General").strip() or "General",
                })
        return items if items else None
    except Exception as e:
        import sys
        print(f"WARNING: PDF AI parse failed, falling back to rule-based: {e}", file=sys.stderr)
        return None


def convert_pdf_to_csv(pdf_path: str, output_path: str = None, original_filename: str = None) -> str:
    """
    Parse a PDF file and generate a CSV file compatible with TaskTastic import.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Optional path for output CSV. Defaults to <pdf_stem>_schedule.csv
                     in the same directory as the PDF.
        original_filename: Optional original filename (e.g. from upload) for course/tag extraction.

    Returns:
        Path to the generated CSV file.

    Raises:
        ImportError: If pdfplumber is not installed (pip install pdfplumber).
        FileNotFoundError: If the PDF file does not exist.
    """
    if not pdfplumber:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_path = output_path or str(pdf_path.parent / (pdf_path.stem + "_schedule.csv"))
    # Use original filename for course extraction when provided (e.g. "M156-Syllabus.pdf")
    filename_for_course = (original_filename or pdf_path.name).strip()

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        tables_data = [(p.page_number, tbl) for p in pdf.pages for tbl in (p.extract_tables() or [])]

    # Format tables for AI (page N: row1 | row2 | ...)
    tables_str = ""
    for page_num, tbl in tables_data:
        if tbl:
            rows = [" | ".join(str(c or "") for c in row) for row in tbl]
            tables_str += f"\nPage {page_num}:\n" + "\n".join(rows) + "\n"

    # Prefer AI parsing when ANTHROPIC_API_KEY is set
    items = None
    if _has_anthropic_key():
        items = _parse_with_ai(text, tables_str, filename_for_course)

    if not items:
        items = _parse_content(text, tables_data, filename_for_course)

    _write_csv(items, output_path)
    print(f"[PDF import] output={output_path} tasks={len(items)}", file=sys.stderr)
    return output_path


def _extract_year_hint(text: str, filename: str) -> int:
    """Extract year from document header, title, or filename when not in date blocks."""
    current_year = datetime.now().year
    # Search header/title (first 1200 chars) and filename
    header = (text or "")[:1200]
    combined = header + " " + (filename or "")

    def valid_year(y: int) -> bool:
        return 1990 <= y <= current_year + 2  # Allow historical syllabi (e.g. Fall 2009)

    # Year range: 2025-2026, Academic Year 2024–2025 (use end year for schedules)
    m = re.search(r"(?:Academic\s+Year\s+)?(20\d{2})[-–]\s*(20\d{2})", combined, re.I)
    if m:
        y = int(m.group(2))
        if valid_year(y):
            return y

    # Fall/Spring/Summer/Winter 2026, Academic Year 2026
    m = re.search(r"(?:Fall|Spring|Summer|Winter|Academic\s+Year)\s+(20\d{2})", combined, re.I)
    if m:
        y = int(m.group(1))
        if valid_year(y):
            return y

    # Calendar/Schedule/Syllabus 2026 or 2026 Calendar
    m = re.search(r"(?:Calendar|Schedule|Syllabus|Course)\s+(20\d{2})", combined, re.I)
    if m:
        y = int(m.group(1))
        if valid_year(y):
            return y
    m = re.search(r"(20\d{2})\s+(?:Calendar|Schedule|Syllabus)", combined, re.I)
    if m:
        y = int(m.group(1))
        if valid_year(y):
            return y

    # Filename: "M156 Spring 2026", "Fall 2009", "151-1-Morrison"
    m = re.search(r"(?:Spring|Fall|Summer|Winter)\s+(20\d{2})", combined, re.I)
    if m:
        y = int(m.group(1))
        if valid_year(y):
            return y

    # Standalone 20xx near document start (title area)
    m = re.search(r"\b(20\d{2})\b", header[:500])
    if m:
        y = int(m.group(1))
        if valid_year(y):
            return y

    return current_year


def _parse_content(text: str, tables_data: list, filename: str) -> list:
    """Extract structured items (task, date, course) from text and tables."""
    year_hint = _extract_year_hint(text, filename)

    course = _detect_course(text, filename)
    items = []
    seen = set()

    _noise = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
              "week", "topic", "readings", "date", "assessment", "homework due", "lab",
              "keew", "mon,", "tue,", "wed,", "thu,", "fri,", "sat,", "sun,",
              "wa due:", "wa due", "link to this page:",
              "course topics", "syllabus", "suggested textbook problems", "graphing utility: desmos",
              "note schedule subject to change", "sign up for makeup lab if needed; see email."}

    def add(task: str, date: str):
        t = re.sub(r"^(?:readings?|topic|assessment|homework|lab)\s*[:\-]\s*", "", (task or "").strip(), flags=re.I)
        if not t or len(t) < 2 or t.lower().startswith("http") or re.match(r"^page\s+\d+$", t, re.I):
            return
        tl = t.lower().strip().rstrip(",")
        if tl in _noise:
            return
        if tl.endswith(":") and len(tl) < 15:  # e.g. "wa due:", "wvu lab 1:"
            return
        if re.match(r"^(mon|tue|wed|thu|fri|sat|sun),?$", t.strip(), re.I):  # date fragment
            return
        if re.match(r"^\d+\s*keew\s*$", t.strip(), re.I):  # week label e.g. "1 KEEW"
            return
        if tl.startswith("note schedule") or "schedule subject to change" in tl:
            return
        if re.match(r"^\d+$", t.strip()):  # lone week number
            return
        key = (t.lower(), date or "")
        if key in seen:
            return
        seen.add(key)
        items.append({"task": t, "date": date, "course": course})

    date_re = re.compile(
        r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?",
        re.I,
    )
    due_re = re.compile(
        r"(?:due|exam|deadline)\s*(?:by|date)?\s*[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?)",
        re.I,
    )
    wa_re = re.compile(r"WA\s+due:\s*([^;\n]+)", re.I)

    _month_day_re = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?",
        re.I,
    )

    def norm_date(s: str) -> str:
        if not s or not s.strip():
            return ""
        s = s.strip()
        m = re.search(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b", s)
        if m:
            mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y = 2000 + y if y < 50 else 1900 + y
            if y < 2024:
                y = year_hint
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y}-{mo:02d}-{d:02d}"
        m = re.search(r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\b", s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 2024:
                y = year_hint
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y}-{mo:02d}-{d:02d}"
        m = _month_day_re.search(s)
        if m:
            try:
                y = int(m.group(3)) if m.group(3) else year_hint
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {y}", "%b %d %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return ""

    def _add_cell_as_one_item(cell: str, date_iso: str, add_fn, date_re) -> None:
        """Treat entire cell content as one item. Normalize newlines to space."""
        cell = (cell or "").strip()
        if not cell:
            return
        task = re.sub(r"\s+", " ", cell.replace("\n", " "))
        task = date_re.sub("", task).strip()
        task = re.sub(r"^[:\-\*•]\s*", "", task)
        if task and len(task) > 2:
            add_fn(task, date_iso)

    def _parse_schedule_table(table: list, date_col: int, norm_date_fn, add_fn, date_re) -> None:
        """Parse Physics-style schedule: each content cell = one item."""
        content_cols = [i for i in range(len(table[0] or [])) if i != date_col]
        current = ""
        for row in table[1:]:
            row = [str(c or "").strip() for c in row]
            if date_col < len(row) and row[date_col]:
                current = norm_date_fn(row[date_col]) or current
            if not current:
                continue
            for i in content_cols:
                if i < len(row) and row[i]:
                    _add_cell_as_one_item(row[i], current, add_fn, date_re)

    def _is_date_row(cells: list, start_col: int = 1, end_col: int = 6) -> bool:
        """Check if row looks like date cells (e.g. January 12, January 13...)."""
        count = 0
        for i in range(start_col, min(end_col, len(cells))):
            c = (cells[i] or "").strip()
            if _month_day_re.search(c):
                count += 1
        return count >= 2

    def _parse_calendar_grid(table: list, carried: tuple) -> tuple:
        """Parse M156-style calendar: alternating date rows and content rows, Mon–Fri columns.
        Handles both orderings: dates-before-content (page 1) and content-before-dates (page 2+).
        When header/content span page break: content on new page uses dates from previous page.
        Returns (last_date_row, ended_with_date_row) to carry to next page."""
        carried_date_row, prev_ended_with_date_row = carried if carried else (None, False)
        rows = [[str(c or "").strip() for c in row] for row in table]
        day_cols = list(range(1, 6))  # cols 1–5 = Mon–Fri

        def is_date_row(r: list) -> bool:
            return len(r) >= 6 and _is_date_row(r, 1, 6)

        def is_content_row(r: list) -> bool:
            if len(r) < 6:
                return False
            for i in day_cols:
                if i < len(r):
                    c = (r[i] or "").strip()
                    if c and "KEEW" not in c and c.lower() not in ("monday", "tuesday", "wednesday", "thursday", "friday"):
                        if not _month_day_re.search(c):  # not a date
                            return True
            return False

        def process_content_with_dates(content_row: list, date_row: list) -> None:
            date_vals = [norm_date((date_row[i] or "").strip()) for i in day_cols]
            for col_idx, date_iso in zip(day_cols, date_vals):
                if col_idx >= len(content_row):
                    break
                cell = (content_row[col_idx] or "").strip()
                if not cell or "KEEW" in cell or cell.lower() in ("monday", "tuesday", "wednesday", "thursday", "friday"):
                    continue
                task = re.sub(r"\s+", " ", cell.replace("\n", " "))
                task = date_re.sub("", task).strip()
                task = re.sub(r"^[:\-\*•]\s*", "", task)
                task = re.sub(r"\s*\d+\s*KEEW\s*", "", task, flags=re.I)
                if task and len(task) > 2 and "KEEW" not in task:
                    add(task, date_iso or "")


        date_row = carried_date_row
        last_date_row = carried_date_row
        ended_with_date_row = False
        for r, row in enumerate(rows):
            if len(row) < 6:
                continue
            if is_date_row(row):
                date_row = row
                last_date_row = row
                ended_with_date_row = True
                continue
            if not is_content_row(row):
                ended_with_date_row = False
                continue
            ended_with_date_row = False
            # Content row: use date_row from above, from previous page, or from below
            # Prefer carried when prev page ended with date row (content for those dates is on this page)
            if date_row is not None:
                process_content_with_dates(row, date_row)
            elif prev_ended_with_date_row and carried_date_row is not None:
                process_content_with_dates(row, carried_date_row)
            else:
                next_r = r + 1
                if next_r < len(rows) and is_date_row(rows[next_r]):
                    process_content_with_dates(row, rows[next_r])
        return (last_date_row, ended_with_date_row)

    # Parse tables
    calendar_grid_parsed = False
    carried = None  # (last_date_row, ended_with_date_row)
    schedule_date_col = None
    for _, table in tables_data:
        if not table or len(table) < 2:
            continue
        header_row = [str(c or "").lower().strip() for c in (table[0] or [])]
        header_row_2 = [str(c or "").lower().strip() for c in (table[4] or [])] if len(table) > 4 else []
        all_cells = " ".join(str(c or "").lower() for row in table[:6] for c in (row or []))
        has_day_headers = any(
            d in all_cells for d in ("monday", "tuesday", "wednesday", "thursday", "friday")
        )
        date_col = next((i for i, h in enumerate(header_row) if h and ("date" in h or "due" in h)), -1)
        if date_col < 0:
            for row in table[:5]:
                for i, c in enumerate(row or []):
                    if c and ("date" in str(c).lower() or "due" in str(c).lower()):
                        date_col = i
                        break
                if date_col >= 0:
                    break
        if date_col < 0 and schedule_date_col is not None:
            date_col = schedule_date_col
        # Only detect date column from table cells when NOT a calendar grid.
        # Calendar grids have dates in a ROW (one per day column), not a single date column.
        if date_col < 0 and not has_day_headers:
            for row in table[1:6]:
                for i, c in enumerate(row or []):
                    if c and _month_day_re.search(str(c)):
                        date_col = i
                        break
                if date_col >= 0:
                    break

        if has_day_headers and date_col < 0:
            carried = _parse_calendar_grid(table, carried)
            calendar_grid_parsed = True
            continue

        # Physics-style schedule: Week, Topic, Readings, Date, Assessment, Homework Due, Lab
        header_lower = " ".join(header_row) + " " + " ".join(header_row_2)
        has_schedule_cols = "topic" in header_lower and "date" in header_lower and (
            "homework" in header_lower or "assessment" in header_lower or "lab" in header_lower
        )
        if has_schedule_cols and date_col >= 0:
            schedule_date_col = date_col
            _parse_schedule_table(table, date_col, norm_date, add, date_re)
            calendar_grid_parsed = True
            continue

        current = ""
        for row in table[1:]:
            row = [str(c or "").strip() for c in row]
            if date_col >= 0 and date_col < len(row) and row[date_col]:
                current = norm_date(row[date_col]) or current
            for i, cell in enumerate(row):
                if i != date_col and cell:
                    _add_cell_as_one_item(cell, current, add, date_re)


    # Parse text (skip when calendar grid was parsed - text would duplicate/misalign content)
    if not calendar_grid_parsed:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2 and len(text) > 100:
            lines = [l.strip() for l in re.split(r"\n+|\t", text) if l.strip()]
        current = ""
        for line in lines:
            if due_re.search(line):
                m = due_re.search(line)
                if m:
                    current = norm_date(m.group(1)) or current
            if date_re.search(line):
                m = date_re.search(line)
                if m:
                    current = norm_date(m.group(0)) or current
            for m in wa_re.finditer(line):
                for part in m.group(1).split(","):
                    add(part.strip(), current)
            task_part = date_re.sub("", line)
            task_part = due_re.sub("", task_part)
            task_part = wa_re.sub("", task_part)
            task_part = re.sub(r"\s+", " ", task_part).strip()
            task_part = re.sub(r"^(?:readings?|topic|assessment|homework|lab)\s*[:\-]\s*", "", task_part, flags=re.I)
            if task_part:
                for p in re.split(r"[;•]|(?:\s+and\s+)", task_part):
                    p = p.strip()
                    if p:
                        add(p, current)

    return items


def _detect_course(text: str, filename: str) -> str:
    """Extract course name from PDF content or filename."""
    first = (text or "")[:1200]
    m = re.search(r"(?:course|syllabus)\s*(?:title|name|number)?\s*[:\-]\s*([^\n]{3,80})", first, re.I)
    if m:
        name = m.group(1).strip()
        return re.sub(r"\s+", " ", name)
    m = re.search(r"(?:Physics|PHYS|Math|MATH|M\d{2,4}|CS|History|HIST|ENG|BIO)\s*\d{2,4}[A-Z]?", text or "", re.I)
    if m:
        return m.group(0).strip()
    m = re.search(r"[A-Z]{2,6}\s*\d{3}[A-Z]?", text or "")
    if m:
        return m.group(0).strip()
    # From filename: extract course name or use filename stem (without extension)
    fn = (filename or "").replace("Copy of ", "").replace(" - Sheet1", "").strip()
    # Remove extension for stem (e.g. "M156-Syllabus.pdf" -> "M156-Syllabus")
    stem = re.sub(r"\.(pdf|csv)$", "", fn, flags=re.I).strip() if fn else ""
    # Try known course patterns in filename
    m = re.search(r"(Physics\s*111|M\d{2,4}|MATH\s*\d{2,4}|[A-Z]{2,6}\s*\d{3}[A-Z]?)", stem or fn, re.I)
    if m:
        return m.group(1).strip()
    # Fallback: use filename stem as course/tag (e.g. "M156 Spring 2026", "Physics 101 Syllabus")
    generic = ("", "syllabus", "schedule", "calendar", "document", "import", "pdf_import")
    if stem and stem.lower() not in generic and len(stem) < 80:
        return re.sub(r"\s+", " ", stem)
    return "General"


def _format_mdyyyy(iso: str) -> str:
    """Convert YYYY-MM-DD to M/D/YYYY for CSV display."""
    if not iso or not re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
        return ""
    p = iso.split("-")
    return f"{int(p[1])}/{int(p[2])}/{p[0]}"


def _write_csv(items: list, output_path: str) -> None:
    """Write items to CSV in TaskTastic import format."""
    def esc(s: str) -> str:
        s = (s or "").strip()
        if "," in s or '"' in s or "\n" in s:
            return '"' + s.replace('"', '""') + '"'
        return s

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write("Category,Tag,Task,Due Date,Status\n")
        for item in items:
            course = item.get("course", "General")
            task = (item.get("task") or "").strip()
            due = _format_mdyyyy(item.get("date", "") or "")
            row = [esc("School"), esc(course), esc(task), esc(due), esc("Open")]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_csv.py <input.pdf> [output.csv] [original_filename]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Parses a PDF (syllabus, schedule) and outputs CSV for TaskTastic import.", file=sys.stderr)
        print("Output format: Category,Tag,Task,Due Date,Status", file=sys.stderr)
        sys.exit(1)
    pdf_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    original_filename = sys.argv[3] if len(sys.argv) > 3 else None
    try:
        result = convert_pdf_to_csv(pdf_path, out_path, original_filename)
        print(f"Wrote schedule to {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
