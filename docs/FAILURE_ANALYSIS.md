# What citation checks missed

The initial Granite 4.1 8B workflow produced a schema-valid memo with seven valid source references. Its limitations nevertheless claimed that the filings did not provide specific LCR or NSFR values. Bank of America's 2025 annual filing reports an average consolidated LCR of 112% and an average consolidated NSFR of 120% for the three months ended December 31, 2025. [Original report](https://investor.bankofamerica.com/regulatory-and-other-filings/select-sec-filings/content/0000070858-26-000157/bac-20251231.htm)

Two problems interacted. The first HTML parser collected paragraphs and tables but missed most inline-XBRL narrative stored in div elements. The model then inferred absence from the full filing using incomplete retrieved passages. A source-ID membership check did not check that uncited limitation statement.

The corrected parser includes leaf div paragraphs without duplicating ancestors; a regression test covers this case. The revised prompt preserves quarterly-average time bases and restricts absence statements to the supplied passages. The initial 13,965-block snapshot remains local for audit; the corrected corpus contains 26,058 blocks. Comparison summaries separate source snapshots and workflow versions. These changes have not been assigned a fabricated accuracy improvement.

`evals/reviews.json` records this agent spot review, its exact run identity and counterevidence IDs. It is not an independent human gold annotation. Claim entailment remains unmeasured until an appropriate evaluation set is labeled.

Other initial runtime findings: Qwen and MiMo processors needed Torchvision; Spark's custom attention could not operate on packed Byte weights; Mistral's nested tokenizer did not expose the padding property used by the adapter. The corrected profiles add the vision dependency, use FP16 Spark weights and leave generation padding to the model configuration. Initial generation failures remain in LangSmith rather than being presented as successful generations; setup errors are documented separately.
