var server = require('http').createServer();
var io = require('socket.io')(server);

io.on('connection', (client) => {
  client.on('event', (data) => {
    console.log('data:', data);
  });
  client.on('disconnect', () => {});
});

server.listen(3000);


