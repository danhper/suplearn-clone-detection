CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY,
  contest_id INTEGER,
  contest_type VARCHAR(64),
  problem_id INTEGER,
  problem_title VARCHAR(255),
  language VARCHAR(64),
  language_code VARCHAR(64),
  source_length INTEGER,
  exec_time INTEGER,
  tokens_count INTEGER,
  ast TEXT,
  url VARCHAR(255),
  FOREIGN KEY (problem_id) REFERENCES problems (id),
  FOREIGN KEY (contest_id) REFERENCES contests (id)
);

CREATE INDEX IF NOT EXISTS contest_idx ON submissions (contest_id, contest_type);
CREATE INDEX IF NOT EXISTS problem_idx ON submissions (contest_id, contest_type, problem_id);
CREATE INDEX IF NOT EXISTS language_idx ON submissions (language_code);

CREATE VIEW IF NOT EXISTS submissions_stats AS
  SELECT id, contest_id, contest_type, problem_id, problem_title,
  language, language_code, source_length, exec_time, tokens_count, url
  FROM submissions;
