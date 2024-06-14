import asyncio

async def main():
	# using future
	future = asyncio.get_running_loop().create_future()
	future.add_done_callback(lambda r: print("future done callback"))
	future.set_result("done")
	print("future processed")
	await future
	print("future awaited (1)")
	await future
	print("future awaited (2)")
	
	# using task
	async def coroutine():
		print("task processed")
	task = asyncio.create_task(coroutine())
	task.add_done_callback(lambda r: print("task done callback"))
	await task
	print("task awaited (1)")
	await task
	print("task awaited (2)")

asyncio.run(main())
