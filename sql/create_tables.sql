CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY,
  contest_id INTEGER NOT NULL,
  contest_type VARCHAR(64) NOT NULL,
  problem_id INTEGER NOT NULL,
  problem_title VARCHAR(255),
  filename VARCHAR(255) NOT NULL,
  language VARCHAR(64),
  language_code VARCHAR(64) NOT NULL,
  source_length INTEGER,
  exec_time INTEGER,
  tokens_count INTEGER,
  ast TEXT NOT NULL,
  url VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS contest_idx ON submissions (contest_id, contest_type);
CREATE INDEX IF NOT EXISTS problem_idx ON submissions (contest_id, contest_type, problem_id);
CREATE INDEX IF NOT EXISTS language_idx ON submissions (language_code);
CREATE INDEX IF NOT EXISTS token_count_idx ON submissions (tokens_count);
CREATE INDEX IF NOT EXISTS token_count_idx ON submissions (tokens_count, language_code);

CREATE VIEW IF NOT EXISTS submissions_stats AS
  SELECT id, contest_id, contest_type, problem_id, problem_title, filename,
  language, language_code, source_length, exec_time, tokens_count, url
  FROM submissions;

CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY,

  anchor_id INTEGER NOT NULL,
  positive_id INTEGER,
  negative_id INTEGER,

  set_name STRING NOT NULL,
  config_checksum STRING NOT NULL,

  FOREIGN KEY (anchor_id) REFERENCES submissions (id),
  FOREIGN KEY (positive_id) REFERENCES submissions (id),
  FOREIGN KEY (negative_id) REFERENCES submissions (id)
);

CREATE INDEX IF NOT EXISTS set_name_idx ON samples (set_name);
CREATE INDEX IF NOT EXISTS anchor_idx ON samples (anchor_id);
CREATE INDEX IF NOT EXISTS positive_idx ON samples (positive_id);
CREATE INDEX IF NOT EXISTS negative_idx ON samples (negative_id);
CREATE INDEX IF NOT EXISTS config_checksum_idx ON samples (config_checksum);


-- SELECT * FROM submissions WHERE tokens_count BETWEEN 448 AND 672 AND submissions.language_code = "java" AND (problem_id != 2 OR submissions.contest_id != 35 OR submissions.contest_type != "b");