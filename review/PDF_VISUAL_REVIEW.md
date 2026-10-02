# Final PDF visual QA

**Result: no blocking visual defect observed.** All 26 Reader Guide pages and all 49 Supplement pages were actually checked in the existing contact sheets. Twelve selected pages were additionally opened individually: Reader 3, 4, 21, 23, 26; Supplement 3, 13, 14, 15, 16, 34, 48. The page-by-page receipt distinguishes these inspection levels and binds PDFs, PNGs, contact sheets and extracted text by SHA256.

The review covered visible clipping, overlapping text, blank pages, unresolved references, missing glyphs, overflowing tables, figure/caption alignment and private addresses. No such blocking defect was found. Dense Supplement tables on pages 13–15 and 34 are within their page boundaries and have separated rows/columns. Reader result figures on pages 21 and 23 have readable labels and aligned captions. Chinese glossary pages 3–4 and references on page 26 render correctly in the enlarged views.

One **non-blocking editorial note** remains: Supplement page 16 contains only a short paragraph at the top, with a large unused area before the full-page figure on page 17. It is not empty and no content is cropped. Reader page 25 and Supplement page 49 are similarly short concluding pages, with valid content rather than accidental blank pages. Pagination polish is optional; no edit was made.

Text checks found zero `??` unresolved-reference placeholders, zero Unicode replacement characters and zero matches for the inspected local/private address patterns in both documents. The Supplement's single question mark is ordinary prose on page 3 (“How does an added current change the measured material derivative?”), not a broken citation. Visible citations and mathematical glyphs showed no missing boxes in the reviewed images.

This is mechanical visual QA only, not a renewed scientific, bibliography or privacy audit of linked files. Contact-sheet inspection establishes observed page layout; it does not claim every small glyph on all 75 pages was examined at full size. No PDF, source, data or scientific conclusion was changed, and no authoring marker, re-render or new numerical experiment was run.

Detailed page receipt: `receipt.json`.

## V2 bounded layout recheck

**PASS: the old sparse-page note is resolved.** The new 48-page Supplement matches SHA256 `bab63d990ebbfb16f52426454dfb72cf302f3fadef5306dd64e01b8039860803`. Pages 13–18 and all later pages through 48 were actually inspected in contact sheets 3–8; pages 16, 17, 47 and 48 were additionally viewed individually. Pages 1–12 were not visually rechecked in this bounded delta.

Figure S4 now sits on page 16, with the former introductory text incorporated in its caption. Image, caption, page number and margins are separated; no orphan paragraph remains. The following table/section page 17 and the shifted later pages show no new clipping, overlap, table overflow, blank pages or figure/caption alignment problems. The final pages 47–48 retain the cost table and complete provenance section with no cropped tail. Text scans again show no unresolved `??`, replacement characters or matching private-address patterns.

The old receipt is unchanged. New page-level scope and hashes are in `receipt_v2.json`. This review verifies layout only; numerical/scientific equivalence was not independently re-audited. No draft, source, PDF or experiment was changed.
