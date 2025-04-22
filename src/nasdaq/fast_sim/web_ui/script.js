const socket = io();
const feed = document.getElementById("feed");

socket.on('fast_data', (msg) => {
  const div = document.createElement("div");
  div.className = "msg";
  div.textContent = JSON.stringify(msg, null, 2);
  feed.prepend(div);
});
