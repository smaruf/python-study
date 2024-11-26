import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    """Main entry point for the asynchronous example."""

    # Using future
    future = asyncio.get_running_loop().create_future()
    future.add_done_callback(lambda r: logging.info("Future done callback"))
    future.set_result("done")
    logging.info("Future processed")
    await future
    logging.info("Future awaited")

    # Using task
    async def coroutine():
        logging.info("Task processed")
    task = asyncio.create_task(coroutine())
    task.add_done_callback(lambda r: logging.info("Task done callback"))
    await task
    logging.info("Task awaited")

asyncio.run(main())
