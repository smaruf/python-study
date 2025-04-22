const socket = io();
const stream = document.getElementById("stream");

socket.on('fix_data', (msg) => {
  const div = document.createElement("div");
  div.className = "fix";
  div.textContent = JSON.stringify(msg, null, 2);
  stream.prepend(div);
});
