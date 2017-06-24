var server = require('http').createServer();
var io = require('socket.io')(server);

const onConnect = (client) => {
  client.on('UPDATE_SCORE', (data) => {
    console.log('UPDATE_SCORE:', data.score);
    client.broadcast.emit('/', data.score);
  });
  client.on('disconnect', () => {});
}
io.on('connection', onConnect);

server.listen(3000);

