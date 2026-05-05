from app.workers.celery_app import celery_app


class Stage2Dispatcher:
    def dispatch(
        self,
        input_id: str,
        storage_key: str,
        routing_key: str,
        classification_kind: str,
    ) -> None:
        payload = {
            "input_id": input_id,
            "storage_key": storage_key,
            "routing_key": routing_key,
            "classification_kind": classification_kind,
        }

        celery_app.send_task(
            "stage2.process_input",
            args=[payload],
            queue="stage2",
            routing_key=routing_key,
        )