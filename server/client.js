const io = require('socket.io-client');
const socket = io('http://192.168.25.151:3000');

socket.on('connect', () => {
  console.log('connected')
  socket.emit('UPDATE_SCORE', { score: 1 })
});

socket.on('/', (data) => {
  console.log(data)
})

socket.on('disconnect', () => {
  console.log('byebye')
});
