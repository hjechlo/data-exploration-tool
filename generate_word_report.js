/**
 * generate_word_report.js
 *
 * Generates a professional Word (.docx) data dictionary report from the
 * pipeline's JSON payload.
 *
 * Usage
 * -----
 *   node generate_word_report.js <payload.json> <output.docx>
 */

"use strict";
 
const fs = require("fs");
const path = require("path");
 
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  HeadingLevel,
  AlignmentType,
  BorderStyle,
  WidthType,
  ShadingType,
  VerticalAlign,
  PageBreak,
  LevelFormat,
  ExternalHyperlink,
} = require("docx");
 
// ── Helpers ────────────────────────────────────────────────────────────────
 
const BRAND_BLUE   = "2E75B6";
const HEADER_BG    = "D6E4F0";
const ALT_ROW_BG   = "F2F7FC";
const ERROR_BG     = "FFF3CD";
const BORDER_COLOR = "BBCFE0";
const PAGE_WIDTH   = 9360; // US Letter, 1-inch margins
 
function cellBorder(color = BORDER_COLOR) {
  const b = { style: BorderStyle.SINGLE, size: 1, color };
  return { top: b, bottom: b, left: b, right: b };
}
 
function headerCell(text, widthDxa, bold = true) {
  return new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    borders: cellBorder(),
    shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold, size: 18, font: "Arial" })],
      }),
    ],
  });
}
 
function dataCell(text, widthDxa, shade = false, italic = false) {
  return new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    borders: cellBorder(),
    shading: shade
      ? { fill: ALT_ROW_BG, type: ShadingType.CLEAR }
      : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({
        children: [
          new TextRun({ text: String(text || ""), size: 18, font: "Arial", italics: italic }),
        ],
        spacing: { line: 360 }
      }),
    ],
  });
}
 
function errorCell(text, widthDxa) {
  return new TableCell({
    width: { size: widthDxa, type: WidthType.DXA },
    borders: cellBorder(),
    shading: text ? { fill: ERROR_BG, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({
        children: [
          new TextRun({ text: String(text || ""), size: 18, font: "Arial", color: text ? "7B4700" : "000000" }),
        ],
      }),
    ],
  });
}
 
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 32, font: "Arial", color: BRAND_BLUE })],
    spacing: { before: 360, after: 180 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: BRAND_BLUE, space: 4 },
    },
  });
}
 
function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 26, font: "Arial", color: "1A4E78" })],
    spacing: { before: 240, after: 120 },
  });
}
 
function bodyPara(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 20, font: "Arial" })],
    spacing: { after: 100 , line: 360 },
  });
}
 
function spacer() {
  return new Paragraph({ children: [new TextRun("")], spacing: { after: 120 } });
}
 
// ── Cover page ─────────────────────────────────────────────────────────────
 
function buildCoverPage(payload) {
  const tableCount = Object.keys(payload.tables).length;
  const colCount = Object.values(payload.tables).reduce((s, t) => s + t.length, 0);
  const title = payload.report_title || "Data Dictionary Report";
  const subtitle = payload.report_subtitle || "Automated Profiling & Quality Analysis";
 
  return [
    spacer(), spacer(), spacer(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: title, bold: true, size: 56, font: "Arial", color: BRAND_BLUE }),
      ],
      spacing: { after: 240 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: subtitle, size: 28, font: "Arial", color: "555555" }),
      ],
      spacing: { after: 480 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `Generated: ${payload.generated_at}`, size: 22, font: "Arial", color: "777777" })],
      spacing: { after: 120 },
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `${tableCount} table(s)  ·  ${colCount} column(s)`, size: 22, font: "Arial", color: "777777" })],
    }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}
 
// ── Executive summary ──────────────────────────────────────────────────────
 
function buildExecutiveSummary(payload) {
  const children = [heading1("Executive Summary"), spacer()];
 
  const tableCount = Object.keys(payload.tables).length;
  const colCount = Object.values(payload.tables).reduce((s, t) => s + t.length, 0);
  const candidateJoinCount = (payload.candidate_join_paths || payload.join_paths || []).length;
  const classifiedRelationshipCount = (payload.join_paths || []).length;
  const dupeCount = payload.cross_table_duplicates.length;
  const totalErrors = Object.values(payload.dataset_overviews || {})
    .reduce((s, d) => s + (d.columns_with_errors || 0), 0);
  const totalMissing = Object.values(payload.dataset_overviews || {})
    .reduce((s, d) => s + (d.columns_with_missing || 0), 0);
 
  // LLM-generated executive summary prose
  if (payload.report_summary) {
    children.push(bodyPara(payload.report_summary));
    children.push(spacer());
  } else {
    children.push(bodyPara(
      `This report covers ${colCount} columns across ${tableCount} dataset(s). ` +
      `MinHash analysis identified ${candidateJoinCount} candidate join/linkage path(s) and ` +
      `${dupeCount} cross-table duplicate column(s).`
    ));
    children.push(spacer());
  }
 
  // At-a-glance stat highlights bar
  const statWidths = [1560, 1560, 1560, 1560, 1560, 1560];
  children.push(
    new Table({
      width: { size: PAGE_WIDTH, type: WidthType.DXA },
      columnWidths: statWidths,
      rows: [
        new TableRow({
          children: [
            headerCell("Datasets", statWidths[0]),
            headerCell("Columns", statWidths[1]),
            headerCell("Columns with Quality Notes", statWidths[2]),
            headerCell("Columns with Missing", statWidths[3]),
            headerCell("Candidate Paths", statWidths[4]),
            headerCell("Classified Signals", statWidths[5]),
          ],
        }),
        new TableRow({
          children: [
            dataCell(String(tableCount), statWidths[0]),
            dataCell(String(colCount), statWidths[1]),
            dataCell(String(totalErrors), statWidths[2]),
            dataCell(String(totalMissing), statWidths[3]),
            dataCell(String(candidateJoinCount), statWidths[4]),
            dataCell(String(classifiedRelationshipCount), statWidths[5]),
          ],
        }),
      ],
    })
  );
  children.push(spacer());
 
  // ── Dataset overviews ───────────────────────────────────────────────
  if (payload.dataset_overviews && Object.keys(payload.dataset_overviews).length > 0) {
    children.push(heading2("Dataset Overviews"));
 
    for (const [tableName, overview] of Object.entries(payload.dataset_overviews)) {
      // Dataset name header
      children.push(
        new Paragraph({
          children: [
            new TextRun({ text: tableName, bold: true, size: 22, font: "Arial", color: BRAND_BLUE }),
          ],
          spacing: { before: 200, after: 80 },
          border: {
            bottom: { style: BorderStyle.SINGLE, size: 4, color: "BBCFE0", space: 4 },
          },
        })
      );
 
      // LLM-generated summary paragraph
      if (overview.summary) {
        children.push(bodyPara(overview.summary));
        children.push(spacer());
      }
 
      // Stats mini-table
      const statWidths = [2400, 2400, 2400, 2160];
      const typeStr = Object.entries(overview.type_breakdown || {})
        .map(([t, n]) => `${t}: ${n}`)
        .join("  |  ");
 
      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: statWidths,
          rows: [
            new TableRow({
              children: [
                headerCell("Total Columns", statWidths[0]),
                headerCell("Columns with Quality Notes", statWidths[1]),
                headerCell("Columns with Missing Values", statWidths[2]),
                headerCell("Data Type Breakdown", statWidths[3]),
              ],
            }),
            new TableRow({
              children: [
                dataCell(String(overview.total_columns), statWidths[0]),
                dataCell(String(overview.columns_with_errors), statWidths[1]),
                dataCell(String(overview.columns_with_missing), statWidths[2]),
                dataCell(typeStr || "—", statWidths[3]),
              ],
            }),
          ],
        })
      );
      children.push(spacer());
    }
  }
 
  // Join paths table
  const possibleJoinPaths = payload.candidate_join_paths || payload.join_paths || [];
  const classifiedJoinPaths = payload.join_paths || [];

  const relationships = payload.relationships || {
    foreign_keys: [],
    lookup_tables: [],
    one_to_one_keys: [],
    shared_join_keys: [],
    shared_value_domains: [],
    primary_keys: {},
  };

  const hasRelationshipContent =
    possibleJoinPaths.length > 0 ||
    classifiedJoinPaths.length > 0 ||
    (relationships.foreign_keys && relationships.foreign_keys.length > 0) ||
    (relationships.one_to_one_keys && relationships.one_to_one_keys.length > 0) ||
    (relationships.shared_join_keys && relationships.shared_join_keys.length > 0) ||
    (relationships.shared_value_domains && relationships.shared_value_domains.length > 0) ||
    (relationships.lookup_tables && relationships.lookup_tables.length > 0);

  if (hasRelationshipContent) {
    children.push(heading2("Join Paths"));

    // === ALL POSSIBLE JOIN PATHS ===
    // This is the broad discovery table from MinHash, shingle, and coverage signals.
    // It intentionally includes possible linkage candidates that are not confirmed PK/FK relationships.
    if (possibleJoinPaths.length > 0) {
      children.push(
        new Paragraph({
          text: "POSSIBLE CANDIDATE JOIN / LINKAGE PATHS",
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 200, after: 100 },
          thematicBreak: true,
        })
      );

      const joinColWidths = [2400, 2400, 1320, 1620, 1620];
      const joinHeaderRow = new TableRow({
        children: [
          headerCell("Table A.Column", joinColWidths[0]),
          headerCell("Table B.Column", joinColWidths[1]),
          headerCell("Exact MinHash", joinColWidths[3]),
          headerCell("Shingle", joinColWidths[4]),
        ],
      });

      const fmtScore = (v) => {
        if (v === null || v === undefined || v === "") return "—";
        if (typeof v === "number") return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
        return String(v);
      };

      const joinDataRows = possibleJoinPaths.map((jp, idx) => {

      return new TableRow({
        children: [
          dataCell(`${jp.table_a}.${jp.col_a}`, joinColWidths[0], idx % 2 !== 0),
          dataCell(`${jp.table_b}.${jp.col_b}`, joinColWidths[1], idx % 2 !== 0),
          dataCell(fmtScore(jp.resemblance ?? jp.jaccard), joinColWidths[3], idx % 2 !== 0),
          dataCell(fmtScore(jp.resemblance_shingle), joinColWidths[4], idx % 2 !== 0),
        ],
      });
    });

      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: joinColWidths,
          rows: [joinHeaderRow, ...joinDataRows],
        })
      );

      children.push(spacer());
    }

    // === KEY IDENTIFICATION ===
    const hasKeyIdentification =
      (relationships.foreign_keys && relationships.foreign_keys.length > 0) ||
      (relationships.one_to_one_keys && relationships.one_to_one_keys.length > 0) ||
      (relationships.lookup_tables && relationships.lookup_tables.length > 0) ||
      (relationships.shared_join_keys && relationships.shared_join_keys.length > 0) ||
      (relationships.shared_value_domains && relationships.shared_value_domains.length > 0) ||
      (relationships.primary_keys && Object.keys(relationships.primary_keys).length > 0);

    if (hasKeyIdentification) {
      children.push(
        new Paragraph({
          text: "RELATIONSHIP CLASSIFICATION",
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 240, after: 100 },
          thematicBreak: true,
        })
      );
    }

    // === CANDIDATE PRIMARY KEYS ===
    if (relationships.primary_keys && Object.keys(relationships.primary_keys).length > 0) {
      children.push(
        new Paragraph({
          text: `Likely Primary Keys Referenced by Other Columns (${Object.keys(relationships.primary_keys).length})`,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 160, after: 100 },
        })
      );

      const pkColWidths = [3600, 5760];
      const pkHeaderRow = new TableRow({
        children: [
          headerCell("Primary Key Candidate", pkColWidths[0]),
          headerCell("Referenced By", pkColWidths[1]),
        ],
      });

      const pkDataRows = Object.entries(relationships.primary_keys).map(([pk, fks], idx) =>
        new TableRow({
          children: [
            dataCell(pk, pkColWidths[0], idx % 2 !== 0),
            dataCell(Array.isArray(fks) ? fks.join(" | ") : "—", pkColWidths[1], idx % 2 !== 0),
          ],
        })
      );

      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: pkColWidths,
          rows: [pkHeaderRow, ...pkDataRows],
        })
      );

      children.push(spacer());
    }

    // === PRIMARY / FOREIGN KEY RELATIONSHIPS ===
    if (relationships.foreign_keys && relationships.foreign_keys.length > 0) {
      children.push(
        new Paragraph({
          text: `Primary / Foreign Key Relationships (${relationships.foreign_keys.length})`,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 160, after: 100 },
        })
      );

      const fkColWidths = [2800, 2800, 1500, 2260];
      const fkHeaderRow = new TableRow({
        children: [
          headerCell("Primary Key Candidate", fkColWidths[0]),
          headerCell("Foreign Key Candidate", fkColWidths[1]),
          headerCell("Ref. Integrity", fkColWidths[2]),
          headerCell("Interpretation", fkColWidths[3]),
        ],
      });

      const fkDataRows = relationships.foreign_keys.map((fk, idx) => {
        const integrity = fk.referential_integrity || 0;

        return new TableRow({
          children: [
            dataCell(fk.references || "—", fkColWidths[0], idx % 2 !== 0),
            dataCell(fk.foreign_key || "—", fkColWidths[1], idx % 2 !== 0),
            dataCell(`${integrity.toFixed(1)}%`, fkColWidths[2], idx % 2 !== 0),
            dataCell(fk.interpretation || "—", fkColWidths[3], idx % 2 !== 0, true),
          ],
        });
      });

      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: fkColWidths,
          rows: [fkHeaderRow, ...fkDataRows],
        })
      );

      children.push(spacer());
    }

    // === ONE-TO-ONE SHARED KEYS ===
    if (relationships.one_to_one_keys && relationships.one_to_one_keys.length > 0) {
      children.push(
        new Paragraph({
          text: `One-to-One Shared Keys (${relationships.one_to_one_keys.length})`,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 160, after: 100 },
        })
      );

      const keyColWidths = [2600, 2600, 2600, 1560];
      const keyHeaderRow = new TableRow({
        children: [
          headerCell("Key A", keyColWidths[0]),
          headerCell("Key B", keyColWidths[1]),
          headerCell("Interpretation", keyColWidths[2]),
        ],
      });

      const keyDataRows = relationships.one_to_one_keys.map((rel, idx) => {
        const quality = rel.quality_score || 0;

        return new TableRow({
          children: [
            dataCell(rel.key_a || "—", keyColWidths[0], idx % 2 !== 0),
            dataCell(rel.key_b || "—", keyColWidths[1], idx % 2 !== 0),
            dataCell(rel.interpretation || "—", keyColWidths[2], idx % 2 !== 0, true),
          ],
        });
      });

      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: keyColWidths,
          rows: [keyHeaderRow, ...keyDataRows],
        })
      );
      children.push(spacer());
    }

    // === SHARED JOIN KEYS ===
    if (relationships.shared_join_keys && relationships.shared_join_keys.length > 0) {
      children.push(
        new Paragraph({
          text: `Shared Join Keys (${relationships.shared_join_keys.length})`,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 160, after: 100 },
        })
      );
      const sjColWidths = [2800, 2800, 2960, 800];
      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: sjColWidths,
          rows: [
            new TableRow({
              children: [
                headerCell("Column A", sjColWidths[0]),
                headerCell("Column B", sjColWidths[1]),
                headerCell("Interpretation", sjColWidths[2]),
                headerCell("Coverage", sjColWidths[3]),
              ],
            }),
            ...relationships.shared_join_keys.map((rel, idx) =>
              new TableRow({
                children: [
                  dataCell(rel.column_a || "—", sjColWidths[0], idx % 2 !== 0),
                  dataCell(rel.column_b || "—", sjColWidths[1], idx % 2 !== 0),
                  dataCell(rel.interpretation || "—", sjColWidths[2], idx % 2 !== 0, true),
                  dataCell(
                    `${((rel.coverage_a || 0) * 100).toFixed(0)}% / ${((rel.coverage_b || 0) * 100).toFixed(0)}%`,
                    sjColWidths[3], idx % 2 !== 0
                  ),
                ],
              })
            ),
          ],
        })
      );
      children.push(spacer());
    }

        // === SHARED VALUE DOMAINS ===
    if (relationships.shared_value_domains && relationships.shared_value_domains.length > 0) {
      children.push(
        new Paragraph({
          text: `Shared Value Domains (Not Key Joins) (${relationships.shared_value_domains.length})`,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 160, after: 100 },
        })
      );

      const svColWidths = [2800, 2800, 2960, 800];

      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: svColWidths,
          rows: [
            new TableRow({
              children: [
                headerCell("Column A", svColWidths[0]),
                headerCell("Column B", svColWidths[1]),
                headerCell("Interpretation", svColWidths[2]),
                headerCell("Coverage", svColWidths[3]),
              ],
            }),
            ...relationships.shared_value_domains.map((rel, idx) =>
              new TableRow({
                children: [
                  dataCell(rel.column_a || "—", svColWidths[0], idx % 2 !== 0),
                  dataCell(rel.column_b || "—", svColWidths[1], idx % 2 !== 0),
                  dataCell(rel.interpretation || "—", svColWidths[2], idx % 2 !== 0, true),
                  dataCell(
                    `${((rel.coverage_a || 0) * 100).toFixed(0)}% / ${((rel.coverage_b || 0) * 100).toFixed(0)}%`,
                    svColWidths[3],
                    idx % 2 !== 0
                  ),
                ],
              })
            ),
          ],
        })
      );

      children.push(spacer());
    }

    // === LOOKUP TABLES / MANY-TO-MANY ===
    if (relationships.lookup_tables && relationships.lookup_tables.length > 0) {
      children.push(
        new Paragraph({
          text: `Lookup Tables / Many-to-Many (${relationships.lookup_tables.length})`,
          heading: HeadingLevel.HEADING_3,
          spacing: { before: 160, after: 100 },
        })
      );

      const lookupColWidths = [2500, 2500, 2000, 1560];
      const lookupHeaderRow = new TableRow({
        children: [
          headerCell("Column A", lookupColWidths[0]),
          headerCell("Column B", lookupColWidths[1]),
          headerCell("Interpretation", lookupColWidths[2]),
          headerCell("Jaccard", lookupColWidths[3]),
        ],
      });

      const lookupDataRows = relationships.lookup_tables.map((lookup, idx) => {
        const jaccard = lookup.jaccard || 0;

        return new TableRow({
          children: [
            dataCell(lookup.column_a || "—", lookupColWidths[0], idx % 2 !== 0),
            dataCell(lookup.column_b || "—", lookupColWidths[1], idx % 2 !== 0),
            dataCell(lookup.interpretation || "—", lookupColWidths[2], idx % 2 !== 0, true),
            dataCell(`${(jaccard * 100).toFixed(1)}%`, lookupColWidths[3], idx % 2 !== 0),
          ],
        });
      });

      children.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: lookupColWidths,
          rows: [lookupHeaderRow, ...lookupDataRows],
        })
      );
      children.push(spacer());
    }

    if (payload.join_interpretation) {
      children.push(bodyPara(payload.join_interpretation));
      children.push(spacer());
    }

    children.push(spacer());
  }
 
  // Cross-table duplicates table
  if (payload.cross_table_duplicates.length > 0) {
    children.push(heading2("Cross-Table Duplicate Columns (Review Manually)"));
    const colWidths = [3000, 3000, 1680, 1680];
    const headerRow = new TableRow({
      children: [
        headerCell("Table A.Column", colWidths[0]),
        headerCell("Table B.Column", colWidths[1]),
        headerCell("Resemblance", colWidths[2]),
        headerCell("Types", colWidths[3]),
      ],
    });
    const dataRows = payload.cross_table_duplicates.map((dc, idx) =>
      new TableRow({
        children: [
          dataCell(`${dc.table_a}.${dc.col_a}`, colWidths[0], idx % 2 !== 0),
          dataCell(`${dc.table_b}.${dc.col_b}`, colWidths[1], idx % 2 !== 0),
          dataCell(dc.resemblance, colWidths[2], idx % 2 !== 0),
          dataCell(`${dc.type_a} / ${dc.type_b}`, colWidths[3], idx % 2 !== 0),
        ],
      })
    );
    children.push(
      new Table({
        width: { size: PAGE_WIDTH, type: WidthType.DXA },
        columnWidths: colWidths,
        rows: [headerRow, ...dataRows],
      })
    );
    children.push(spacer());
  }


  children.push(new Paragraph({ children: [new PageBreak()] }));
  return children
}

 
// ── Per-column card ────────────────────────────────────────────────────────
 
const CARD_LABEL_W  = 1600;  // left label column
const CARD_CONTENT_W = 7760; // right content column (total = 9360)
 
function labelCell(text) {
  return new TableCell({
    width: { size: CARD_LABEL_W, type: WidthType.DXA },
    borders: cellBorder(),
    shading: { fill: "EEF4FB", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.TOP,
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold: true, size: 17, font: "Arial", color: "1A4E78" })],
        spacing: { line: 360 },
      }),
    ],
  });
}
 
function contentCell(paragraphs) {
  return new TableCell({
    width: { size: CARD_CONTENT_W, type: WidthType.DXA },
    borders: cellBorder(),
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: paragraphs,
  });
}
 
function textPara(text, color = "222222", bold = false) {
  return new Paragraph({
    children: [new TextRun({ text: String(text || ""), size: 18, font: "Arial", color, bold })],
    spacing: { after: 40 , line: 360 },
  });
}
 
function buildColumnCard(col) {
  const rows = [];
 
  // ── Header row: column name + type ──────────────────────────────────
  rows.push(
    new TableRow({
      children: [
        new TableCell({
          columnSpan: 2,
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          borders: cellBorder(BRAND_BLUE),
          shading: { fill: "D6E4F0", type: ShadingType.CLEAR },
          margins: { top: 100, bottom: 100, left: 160, right: 160 },
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: col.column_name, bold: true, size: 22, font: "Arial", color: "1A3A5C" }),
                new TextRun({ text: `   ${col.data_type}`, size: 18, font: "Arial", color: "666666", italics: true }),
              ],
              spacing: { line: 360 },
            }),
          ],
        }),
      ],
    })
  );
 
  // ── Description row ──────────────────────────────────────────────────
  rows.push(
    new TableRow({
      children: [
        labelCell("Description"),
        contentCell([textPara(col.description || "—")]),
      ],
    })
  );
 
  // ── Errors row (only if there are errors) ───────────────────────────
  if (col.errors && col.errors.length > 0) {
    const errorParas = col.errors.map(e =>
      new Paragraph({
        children: [
          new TextRun({ text: "• ", size: 18, font: "Arial", color: "7B4700", bold: true }),
          new TextRun({ text: e, size: 18, font: "Arial", color: "7B4700" }),
        ],
        spacing: { after: 40, line: 360 },
      })
    );
    rows.push(
      new TableRow({
        children: [
          new TableCell({
            width: { size: CARD_LABEL_W, type: WidthType.DXA },
            borders: cellBorder(),
            shading: { fill: "FFF3CD", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            verticalAlign: VerticalAlign.TOP,
            children: [
              new Paragraph({
                children: [new TextRun({ text: "⚠ Errors", bold: true, size: 17, font: "Arial", color: "7B4700" })],
                spacing: { line: 360 },
              }),
            ],
          }),
          new TableCell({
            width: { size: CARD_CONTENT_W, type: WidthType.DXA },
            borders: cellBorder(),
            shading: { fill: "FFFDF0", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            children: errorParas,
          }),
        ],
      })
    );
  }
 
  // ── Recommended actions row ──────────────────────────────────────────
  const actions = col.recommended_actions && col.recommended_actions.length > 0
    ? col.recommended_actions
    : ["No immediate action needed."];
 
  const actionParas = actions.map((a, i) =>
    new Paragraph({
      children: [
        new TextRun({ text: `${i + 1}.  `, size: 18, font: "Arial", color: "1A4E78", bold: true }),
        new TextRun({ text: a, size: 18, font: "Arial", color: "222222" }),
      ],
      spacing: { after: 50, line: 360 },
    })
  );
 
  rows.push(
    new TableRow({
      children: [
        labelCell("Actions"),
        contentCell(actionParas),
      ],
    })
  );
 
  // ── Sample values row ────────────────────────────────────────────────
  if (col.sample_values && col.sample_values.length > 0) {
    rows.push(
      new TableRow({
        children: [
          labelCell("Sample values"),
          contentCell([textPara(col.sample_values.join("  |  "), "555555")]),
        ],
      })
    );
  }
 
  // ── Missing values row ───────────────────────────────────────────────
  if (col.profile && col.profile.missing_count > 0) {
    const pct = (col.profile.missing_pct * 100).toFixed(2);
    rows.push(
      new TableRow({
        children: [
          labelCell("Missing"),
          contentCell([textPara(`${col.profile.missing_count} values (${pct}%)`, "888888")]),
        ],
      })
    );
  }
 
  return new Table({
    width: { size: PAGE_WIDTH, type: WidthType.DXA },
    columnWidths: [CARD_LABEL_W, CARD_CONTENT_W],
    rows,
  });
}
 
// ── Per-table data dictionary ──────────────────────────────────────────────
 
function buildTableSection(tableName, columns, payload) {
  const children = [
    heading1(`Table: ${tableName}`),
    bodyPara(`${columns.length} column(s) documented below.`),
    spacer(),
  ];
 
  for (const col of columns) {
    children.push(buildColumnCard(col));
    children.push(spacer());
  }
 
  // ── Validation rules ───────────────────────────────────────────────
  const rules = (payload.validation_rules || {})[tableName] || [];
  if (rules.length > 0) {
    children.push(heading2(`Validation Rules — ${tableName}`));
    children.push(bodyPara(
      "The following rules are inferred from the column evidence and should hold for every record on future data loads."
    ));
    children.push(spacer());

    const ruleWidths = [600, 2200, 6560];
    children.push(
      new Table({
        width: { size: PAGE_WIDTH, type: WidthType.DXA },
        columnWidths: ruleWidths,
        rows: [
          new TableRow({
            tableHeader: true,
            children: [
              headerCell("#", ruleWidths[0]),
              headerCell("Column", ruleWidths[1]),
              headerCell("Rule", ruleWidths[2]),
            ],
          }),
          ...rules.map((r, idx) =>
            new TableRow({
              children: [
                dataCell(String(r.rule_id ?? idx + 1), ruleWidths[0], idx % 2 !== 0),
                dataCell(r.column || "—", ruleWidths[1], idx % 2 !== 0),
                dataCell(r.rule || "—", ruleWidths[2], idx % 2 !== 0),
              ],
            })
          ),
        ],
      })
    );
    children.push(spacer());
  }

  children.push(new Paragraph({ children: [new PageBreak()] }));
  return children;
}

 
// ── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const [, , payloadPath, outputPath] = process.argv;

  if (!payloadPath || !outputPath) {
    console.error("Usage: node generate_word_report.js <payload.json> <output.docx>");
    process.exit(1);
  }

  const payload = JSON.parse(fs.readFileSync(payloadPath, "utf8"));

  const allChildren = [
    ...buildCoverPage(payload),
    ...buildExecutiveSummary(payload),
  ];

  for (const [tableName, columns] of Object.entries(payload.tables)) {
    allChildren.push(...buildTableSection(tableName, columns, payload));
  }

  // === VALIDATION CHECK RESULTS — record-centric view (PDF Point 3) ===
  const checkResults = payload.validation_check_results || {};
  for (const [tableName, results] of Object.entries(checkResults)) {
    if (!results || !results.violation_records) continue;
    const violationRecords = results.violation_records;
    const totalFailing = results.total_failing_records || 0;

    allChildren.push(
      new Paragraph({
        text: `Records that fail validation — ${tableName}`,
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 400, after: 160 },
      })
    );

    if (violationRecords.length === 0) {
      allChildren.push(
        new Paragraph({ text: "All records pass all validation rules.", spacing: { before: 80, after: 160 } })
      );
      continue;
    }

    const preferredCols = [
      "Row",
      "Record Identifier",
      "Failed Column",
      "Failed Value",
      "Validation Rules Failed",
    ];

    const remainingCols = Object.keys(violationRecords[0])
      .filter(c => !preferredCols.includes(c));

    const cols = [
      ...preferredCols.filter(c => Object.prototype.hasOwnProperty.call(violationRecords[0], c)),
      ...remainingCols,
    ];

    const colWidths = cols.map(c => {
      if (c === "Validation Rules Failed") return 3760;
      if (c === "Failed Value") return 2000;
      if (c === "Record Identifier") return 1600;
      if (c === "Failed Column") return 1400;
      if (c === "Row") return 600;
      return 1200;
    });

    allChildren.push(
        new Table({
          width: { size: PAGE_WIDTH, type: WidthType.DXA },
          columnWidths: colWidths,
          rows: [
            new TableRow({ children: cols.map((c, i) => headerCell(c, colWidths[i])) }),
            ...violationRecords.map((rec, idx) =>
              new TableRow({
                children: cols.map((c, i) => {
                  const listCols = ["Failed Column", "Failed Value", "Validation Rules Failed"];

                  if (listCols.includes(c)) {
                    // Dynamically choose delimiter: 
                    // Rules use "|" 
                    // Failed Column uses ", "
                    // Failed Value uses " | "
                    let delimiter;
                    if (c === "Validation Rules Failed") delimiter = "|";
                    else if (c === "Failed Column") delimiter = ", ";
                    else delimiter = " | "; 
                    
                    const splitItems = String(rec[c] || "").split(delimiter);
                    const items = splitItems.map(item => {
                      // Only bullet when there are multiple items — a single
                      // value doesn't need a bullet marker.
                      const shouldAddBullet = (c !== "Validation Rules Failed") && splitItems.length > 1;
                      const textContent = shouldAddBullet ? "• " + item.trim() : item.trim();
                      
                      return new Paragraph({
                        children: [
                          new TextRun({ text: textContent, size: 18, font: "Arial" })
                        ],
                        spacing: { after: 60 } // Maintains vertical white space
                      });
                    });
                    return new TableCell({
                      width: { size: colWidths[i], type: WidthType.DXA },
                      borders: cellBorder(),
                      margins: { top: 80, bottom: 80, left: 120, right: 120 },
                      shading: idx % 2 !== 0 ? { fill: ALT_ROW_BG, type: ShadingType.CLEAR } : undefined,
                      children: items,
                    });
                  }

                  // DEFAULT: Standard rendering for Row and Record Identifier
                  return dataCell(String(rec[c] ?? "—"), colWidths[i], idx % 2 !== 0);
                })
              })
            ),
          ],
        }),
        new Paragraph({ children: [new TextRun("")], spacing: { after: 120 } }),
        new Paragraph({
          text: `Number of records that fail the rules: ${totalFailing}`,
          spacing: { before: 100, after: 200 },
        })
      );
  }

  const doc = new Document({
    styles: {
      default: {
        document: { run: { font: "Arial", size: 20 } },
        paragraph: { spacing: { line: 360 } },
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
          run: { size: 32, bold: true, font: "Arial", color: BRAND_BLUE },
          paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 },
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
          run: { size: 26, bold: true, font: "Arial", color: "1A4E78" },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 },
        },
      ],
    },
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [{
            level: 0, format: LevelFormat.BULLET, text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          }],
        },
      ],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: allChildren,
    }],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(outputPath, buffer);
  console.log(`Report written to: ${outputPath}`);
}

main().catch((err) => {
  console.error("Error generating report:", err);
  process.exit(1);
});