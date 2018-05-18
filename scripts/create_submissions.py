import json
from os import path
from typing import List

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from suplearn_clone_detection.config import Config
from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection import entities, util


SQL_PATH = path.realpath(path.join(path.dirname(path.dirname(__file__)), "sql"))
KNOWN_LANGUAGES = ["java", "python"]


class SubmissionCreator:
    def __init__(self, config: Config, engine, known_languages: List[str] = None):
        if known_languages is None:
            known_languages = KNOWN_LANGUAGES
        self.config = config
        self.engine = engine
        self.session_maker = sessionmaker(bind=engine)
        self.known_languages = known_languages
        self.ast_loader = ASTLoader(
            config.generator.asts_path,
            config.generator.filenames_path,
            config.generator.file_format
        )

    def normalize_language(self, language):
        for known_lang in self.known_languages:
            if language.startswith(known_lang):
                return known_lang
        return None

    def make_submission(self, submission_obj):
        ast = self.ast_loader.get_ast(submission_obj["file"])
        return entities.Submission(
            id=submission_obj["id"],
            contest_id=submission_obj["contest_id"],
            contest_type=submission_obj["contest_type"],
            problem_id=submission_obj["problem_id"],
            problem_title=submission_obj["problem_title"],
            filename=path.basename(submission_obj["file"]),
            language=submission_obj["language"],
            language_code=self.normalize_language(submission_obj["language"]),
            source_length=submission_obj["source_length"],
            exec_time=submission_obj["exec_time"],
            tokens_count=len(ast),
            ast=json.dumps(ast),
            url=submission_obj["submission_url"],
        )

    def load_submissions(self):
        submissions = []
        with open(self.config.generator.submissions_path, "r") as f:
            for submission_obj in json.load(f):
                if self.ast_loader.has_file(submission_obj["file"]):
                    submissions.append(self.make_submission(submission_obj))
        return submissions

    def create_db(self):
        with self.engine.begin() as conn, \
             open(path.join(SQL_PATH, "create_tables.sql")) as f:
            conn.connection.connection.executescript(f.read())

    def create_submission(self):
        submissions = self.load_submissions()
        with util.session_scope(self.session_maker) as sess:
            sess.bulk_save_objects(submissions)


def main():
    config = Config.from_file("./config.yml")
    engine = create_engine(config.generator.db_path)
    creator = SubmissionCreator(config, engine)
    creator.create_db()
    creator.create_submission()


if __name__ == '__main__':
    main()
