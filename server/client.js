const io = require('socket.io-client');
const socket = io('https://facematch-server.herokuapp.com/');

socket.on('connect', () => {
  console.log('connected')
  socket.emit('event', 'helloooo')
});

socket.on('disconnect', () => {
  console.log('byebye')
});
