import json
from os import path
from typing import List

from suplearn_clone_detection.config import Config
from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection import entities, database
from suplearn_clone_detection.database import Session


SQL_PATH = path.realpath(path.join(path.dirname(path.dirname(__file__)), "sql"))
KNOWN_LANGUAGES = ["java", "python"]


class SubmissionCreator:
    def __init__(self, config: Config, known_languages: List[str] = None):
        if known_languages is None:
            known_languages = KNOWN_LANGUAGES
        self.config = config
        self.submissions_dir = path.dirname(self.config.generator.submissions_path)
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

    def get_source(self, submission_obj):
        filepath = path.join(self.submissions_dir, submission_obj["file"])
        with open(filepath) as f:
            return f.read()

    def make_submission(self, submission_obj):
        ast = self.ast_loader.get_ast(submission_obj["file"])
        return entities.Submission(
            id=submission_obj["id"],
            url=submission_obj["submission_url"],
            contest_type=submission_obj["contest_type"],
            contest_id=submission_obj["contest_id"],
            problem_id=submission_obj["problem_id"],
            problem_title=submission_obj["problem_title"],
            filename=path.basename(submission_obj["file"]),
            language=submission_obj["language"],
            language_code=self.normalize_language(submission_obj["language"]),
            source_length=submission_obj["source_length"],
            exec_time=submission_obj["exec_time"],
            tokens_count=len(ast),
            source=self.get_source(submission_obj),
            ast=json.dumps(ast),
        )

    def load_submissions(self):
        submissions = []
        with open(self.config.generator.submissions_path, "r") as f:
            for submission_obj in json.load(f):
                if self.ast_loader.has_file(submission_obj["file"]):
                    submissions.append(self.make_submission(submission_obj))
        return submissions

    @staticmethod
    def create_db():
        sqlite_conn = Session.connection().connection.connection
        with open(path.join(SQL_PATH, "create_tables.sql")) as f:
            sqlite_conn.executescript(f.read())
            Session.commit()

    def create_submission(self):
        submissions = self.load_submissions()
        Session.bulk_save_objects(submissions)
        Session.commit()


def main():
    config = Config.from_file("./config.yml")
    database.bind_db(config.generator.db_path)
    creator = SubmissionCreator(config)
    creator.create_db()
    creator.create_submission()


if __name__ == '__main__':
    main()
