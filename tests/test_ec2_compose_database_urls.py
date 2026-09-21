from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class Ec2ComposeDatabaseUrlTest(unittest.TestCase):
    def test_ec2_compose_uses_a_url_encoded_database_password(self) -> None:
        """Reserved password characters must not change a PostgreSQL URL's host."""
        for compose_file in (
            REPOSITORY / "deploy/ec2/state-host.compose.yaml",
            REPOSITORY / "deploy/ec2/app-host.compose.yaml",
        ):
            with self.subTest(compose_file=compose_file):
                compose = compose_file.read_text(encoding="utf-8")
                self.assertIn("NEWSLENS_POSTGRES_PASSWORD_URLENCODED", compose)
                self.assertNotIn(
                    "postgresql://newslens:${NEWSLENS_POSTGRES_PASSWORD}@", compose
                )
