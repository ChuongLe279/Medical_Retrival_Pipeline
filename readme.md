vitel-ai-race-project/
│
├── README.md                          # Tổng quan project, hướng dẫn cài đặt (dùng cho BTC review)
├── .gitignore
├── requirements.txt
├── LICENSE                            # nếu cần (tuỳ luật cuộc thi)
│
├── data/
│   ├── raw/
│   │   ├── rxnorm/
│   │   │   ├── RXNCONSO.RRF           # concept + tên + term type
│   │   │   ├── RXNSAT.RRF             # attribute (vd NDC crosswalk)
│   │   │   ├── RXNSTY.RRF             # semantic type
│   │   │   ├── RXNREL.RRF             # quan hệ giữa các concept
│   │   │   └── README.md              # ghi rõ nguồn + ngày tải
│   │   └── icd10/
│   │       ├── icd10cm_order_YYYY.txt # order file CMS
│   │       ├── icd10cm_tabular_YYYY.xml
│   │       └── README.md
│   │
│   ├── processed/
│   │   ├── rxnorm_index/
│   │   │   ├── faiss.index            # dense vector index
│   │   │   ├── bm25_index.pkl         # sparse index
│   │   │   └── metadata.db            # SQLite: rxcui, tên, term_type
│   │   └── icd10_index/
│   │       ├── faiss.index
│   │       ├── bm25_index.pkl
│   │       └── metadata.db
│   │
│   ├── synthetic/
│   │   ├── generation_prompts/        # prompt dùng để sinh synthetic data
│   │   ├── batch_001/                 # theo từng đợt sinh, dễ trace lại
│   │   │   ├── input.jsonl
│   │   │   └── output.jsonl
│   │   └── generation_log.md          # model nào, ngày nào, bao nhiêu sample
│   │
│   ├── seed_annotated/
│   │   ├── raw_text/                  # 15-20 file .txt gốc
│   │   ├── annotations/               # N.json tương ứng, annotate thủ công
│   │   └── inter_annotator_notes.md   # ghi chú bất đồng khi 2 người annotate
│   │
│   └── test_public/
│       ├── input/                     # 100 file .txt từ BTC
│       └── (KHÔNG chứa nhãn thật)
│
├── src/
│   ├── ner_assertion/                          # === NGƯỜI A ===
│   │   ├── model/
│   │   │   ├── train_lora.py                   # script fine-tune LoRA
│   │   │   ├── inference.py                    # load model + infer trên 1 văn bản
│   │   │   └── config_lora.yaml
│   │   ├── rules/
│   │   │   ├── vital_signs_parser.py            # xử lý "VS98.3 12987 56 18 99RA"
│   │   │   ├── abbreviation_map.py              # po, qid:prn, q6h, WBC...
│   │   │   └── lab_name_vs_result_splitter.py   # tách TÊN_XÉT_NGHIỆM vs KẾT_QUẢ
│   │   └── eval/
│   │       ├── compute_metrics.py               # precision/recall/F1 theo type
│   │       └── error_analysis.py
│   │
│   ├── retrieval_linking/                      # === NGƯỜI B ===
│   │   ├── build_index/
│   │   │   ├── parse_rxnorm_rrf.py
│   │   │   ├── parse_icd10_cms.py
│   │   │   ├── build_faiss.py
│   │   │   └── build_bm25.py
│   │   ├── embed/
│   │   │   ├── sapbert_wrapper.py
│   │   │   └── embedding_cache.py               # cache để tránh encode lại
│   │   ├── rerank/
│   │   │   ├── llm_rerank.py
│   │   │   └── rerank_prompt_templates.py
│   │   └── translate/
│   │       ├── vi_to_en_entity.py
│   │       └── medical_term_dict.py             # từ điển thuật ngữ hay dùng, tránh gọi LLM mỗi lần
│   │
│   ├── pipeline/
│   │   ├── run.py                               # entry point chạy full pipeline
│   │   ├── merge_ner_and_linking.py             # ghép output A + B
│   │   └── span_alignment.py                    # fix lại position sau khi ghép
│   │
│   ├── validate/
│   │   ├── schema.json                          # JSON schema chính thức theo đề bài
│   │   ├── validate_submission.py
│   │   └── check_position_consistency.py        # kiểm tra position khớp text gốc
│   │
│   └── utils/
│       ├── io_utils.py
│       ├── logging_config.py
│       └── seed_everything.py                    # fix random seed
│
├── scripts/
│   ├── download_rxnorm.sh
│   ├── download_icd10.sh
│   ├── build_index.sh
│   ├── run_pipeline.sh
│   ├── make_submission.sh
│   └── run_eval_on_seed.sh                       # đo chất lượng nội bộ trước khi submit thật
│
├── configs/
│   ├── pipeline.yaml
│   ├── data_sources.yaml                         # ghi rõ version data đã dùng để nộp
│   └── model_paths.yaml
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── docs/
│   ├── schema_intermediate.md                    # schema trung gian A ↔ B
│   ├── annotation_guideline.md
│   ├── decisions_log.md
│   ├── data_sources.md                           # chi tiết cách tải lại RxNorm/ICD-10
│   └── submission_checklist.md                   # checklist trước khi nộp (5 lần/ngày, đừng phí)
│
├── notebooks/
│   ├── eda_test_public.ipynb
│   └── error_analysis.ipynb
│
├── tests/
│   ├── ner_assertion/
│   │   └── test_vital_signs_parser.py
│   ├── retrieval_linking/
│   │   └── test_build_index.py
│   └── pipeline/
│       ├── test_output_format.py
│       └── test_end_to_end_sample.py
│
├── models/
│   ├── ner_model/                                 # LoRA adapter weights
│   └── embedding_model/                           # SapBERT weights (nếu self-host)
│
└── output/
    ├── output/                                    # N.json sinh ra
    │   ├── 1.json
    │   ├── 2.json
    │   └── ...
    └── output.zip                                 # file nộp cuối cùng