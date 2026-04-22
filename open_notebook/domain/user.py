from typing import ClassVar, Optional

from loguru import logger
from passlib.context import CryptContext

from open_notebook.database.repository import repo_query
from open_notebook.domain.base import ObjectModel

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(ObjectModel):
    table_name: ClassVar[str] = "user"
    email: str
    password_hash: str
    name: str
    role: str = "user"

    def verify_password(self, password: str) -> bool:
        return _pwd_context.verify(password, self.password_hash)

    @classmethod
    async def get_by_email(cls, email: str) -> Optional["User"]:
        try:
            result = await repo_query(
                "SELECT * FROM user WHERE email = $email LIMIT 1",
                {"email": email.lower().strip()},
            )
            return cls(**result[0]) if result else None
        except Exception as e:
            logger.error(f"Error fetching user by email: {e}")
            return None

    @classmethod
    async def create_user(
        cls,
        email: str,
        password: str,
        name: str,
        role: str = "user",
    ) -> "User":
        password_hash = _pwd_context.hash(password)
        user = cls(
            email=email.lower().strip(),
            password_hash=password_hash,
            name=name,
            role=role,
        )
        await user.save()
        return user

    def _prepare_save_data(self):
        data = super()._prepare_save_data()
        data.pop("table_name", None)
        return data
