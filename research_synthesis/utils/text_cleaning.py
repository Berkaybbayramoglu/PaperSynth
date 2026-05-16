"""Plain-text cleanup helpers used by the Writer agent.

The LLM occasionally emits markdown, tables, or ALL-CAPS prose despite being
asked for plain text. These helpers normalize the output back to clean,
readable paragraphs.
"""
import re


def clean_markdown(text: str) -> str:
    """Strip markdown/formatting and normalize whitespace.

    Removes bold/italic/heading markers, converts pipe-tables into
    ``Header: Value`` lines, trims em/en dashes, drops list markers at line
    start, and collapses excessive blank lines. Long ALL-CAPS sentences are
    converted to sentence case while short ALL-CAPS lines are kept as
    section headings.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)

    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    lines = text.split("\n")
    cleaned_lines = []
    in_table = False
    table_header = None

    for line in lines:
        if re.match(r"^[\s\|\-:]+$", line):
            continue

        if "|" in line and line.count("|") >= 2:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if not cells:
                continue
            if not in_table:
                in_table = True
                table_header = cells
                continue
            if table_header and len(cells) <= len(table_header):
                for i, cell in enumerate(cells):
                    if i < len(table_header):
                        cleaned_lines.append(f"  • {table_header[i]}: {cell}")
                cleaned_lines.append("")
            else:
                cleaned_lines.append("  • " + ", ".join(cells))
        else:
            in_table = False
            table_header = None
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[`~]+", "", text)
    text = re.sub(r"={3,}", "", text)
    text = re.sub(r"-{3,}(?=\s|$)", "", text)
    text = text.replace("—", "").replace("–", "")
    text = re.sub(r"^\s*[A-Z]\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\u2028", "\n")
    text = text.replace("\u2029", "\n\n")

    normalized = []
    for line in text.split("\n"):
        leading = len(line) - len(line.lstrip())
        content = line.strip()
        if content:
            content = re.sub(r" {2,}", " ", content)
            normalized.append(" " * leading + content)
        else:
            normalized.append("")
    text = "\n".join(normalized)

    text = re.sub(r"\n{3,}", "\n\n", text)

    fixed_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            fixed_lines.append(line)
            continue
        if (
            stripped.startswith("-")
            or stripped.startswith("•")
            or re.match(r"^[A-Z]\.\s", stripped)
            or re.match(r"^\d+\.\s", stripped)
        ):
            fixed_lines.append(line)
            continue
        if not stripped.isupper():
            fixed_lines.append(line)
            continue
        if len(stripped) < 60 and not any(c in stripped for c in ".,:;?!"):
            fixed_lines.append(line)
        else:
            leading = len(line) - len(line.lstrip())
            sentences = re.split(r"([.!?]\s+)", stripped)
            fixed_sentences = []
            for i, part in enumerate(sentences):
                if i % 2 == 0:
                    if part:
                        fixed = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
                        fixed_sentences.append(fixed)
                else:
                    fixed_sentences.append(part)
            fixed_lines.append(" " * leading + "".join(fixed_sentences))
    text = "\n".join(fixed_lines)

    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def quick_clean(text: str) -> str:
    """Lightweight pass used after the LLM returns each chunk.

    Removes the most common leftover markdown without doing the full
    table/ALL-CAPS reformatting in :func:`clean_markdown`.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("—", "").replace("–", "-")
    text = re.sub(r"^\s*[->•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
