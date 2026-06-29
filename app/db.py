"""
Хранение предсказаний в PostgreSQL.

Все операции «отказоустойчивы»: если БД недоступна или DATABASE_URL не задан,
функции просто ничего не делают (или возвращают пустой список), а приложение
продолжает работать. Это сделано специально, чтобы база была улучшением,
а не точкой отказа сервиса.
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    _PG_OK = True
except Exception:
    _PG_OK = False


def _conn():
    """Открыть короткое соединение. Вернёт None, если БД не настроена."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn or not _PG_OK:
        return None
    return psycopg2.connect(dsn, connect_timeout=3)


def init_db():
    """Создать таблицу predictions, если её ещё нет. Вызывается при старте."""
    try:
        conn = _conn()
        if conn is None:
            logger.warning("DATABASE_URL не задан — хранение предсказаний отключено")
            return
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    horizon_detected BOOLEAN,
                    roll_deg DOUBLE PRECISION,
                    pitch_deg DOUBLE PRECISION,
                    sky_ratio DOUBLE PRECISION,
                    land_ratio DOUBLE PRECISION,
                    anomaly BOOLEAN,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        conn.close()
        logger.info("Таблица predictions готова")
    except Exception as e:
        logger.warning("init_db не выполнен (приложение работает без БД): %s", e)


def insert_prediction(filename, horizon_detected, roll_deg, pitch_deg,
                      sky_ratio, land_ratio, anomaly):
    """Записать одно предсказание. При ошибке — просто пропустить."""
    try:
        conn = _conn()
        if conn is None:
            return
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (filename, horizon_detected, roll_deg, pitch_deg,
                     sky_ratio, land_ratio, anomaly)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (filename, horizon_detected, roll_deg, pitch_deg,
                 sky_ratio, land_ratio, anomaly),
            )
        conn.close()
    except Exception as e:
        logger.warning("insert_prediction не выполнен: %s", e)


def get_recent(limit: int = 100):
    """Вернуть последние предсказания (новые сверху)."""
    try:
        conn = _conn()
        if conn is None:
            return []
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT %s", (limit,)
            )
            rows = cur.fetchall()
        conn.close()
        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "filename": r["filename"],
                "horizon_detected": r["horizon_detected"],
                "roll_deg": r["roll_deg"],
                "pitch_deg": r["pitch_deg"],
                "sky_ratio": r["sky_ratio"],
                "land_ratio": r["land_ratio"],
                "anomaly": r["anomaly"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            })
        return out
    except Exception as e:
        logger.warning("get_recent не выполнен: %s", e)
        return []


def clear_all():
    """Очистить историю предсказаний."""
    try:
        conn = _conn()
        if conn is None:
            return
        with conn, conn.cursor() as cur:
            cur.execute("TRUNCATE predictions")
        conn.close()
    except Exception as e:
        logger.warning("clear_all не выполнен: %s", e)
