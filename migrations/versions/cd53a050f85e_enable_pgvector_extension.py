"""enable pgvector extension

Revision ID: cd53a050f85e
Revises: 0bac639420c1
Create Date: 2026-04-28 18:02:16.174143

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cd53a050f85e"
down_revision: str | Sequence[str] | None = "0bac639420c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    pass


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS vector;")

    pass
