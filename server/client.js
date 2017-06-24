const io = require('socket.io-client');
const socket = io('http://localhost:3000');

socket.on('connect', () => {
  console.log('connected')
  socket.emit('event', 'helloooo')
});

socket.on('disconnect', () => {
  console.log('byebye')
});
