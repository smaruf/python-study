const socket = io();
const stream = document.getElementById("stream");

socket.on('itch_data', (msg) => {
  const div = document.createElement("div");
  div.className = "log";
  div.textContent = JSON.stringify(msg, null, 2);
  stream.prepend(div);
});
