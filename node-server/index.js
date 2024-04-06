var http = require('http');
var express = require('express');
var app = express();
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var cors = require('cors');
var passport = require('passport');
var session = require('express-session');
var socketIO = require('socket.io');
var server = http.createServer(app);
var io = socketIO(server, {
	cors: {
		origin: '*',
	},
});
require('dotenv').config();

var connection = require('./utils/connection');

var User = require('./routes/user');
var Video = require('./routes/video');
var Web = require('./routes/web');
var Payment = require('./routes/payment');
var Conversation = require('./routes/conversation');
var ErrorHandler = require('./utils/error');
var errorMiddleware = require('./middleware/errorMiddleware');
var Scheduler = require('./utils/scheduler');

// ? Schedulers
Scheduler.expireBid();
Scheduler.jobChecker();
Scheduler.sendRecommendations();
Scheduler.filterUploadedVideos();

connection.connectDB();

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(cors());
app.use(
	session({
		secret: process.env.SECRET,
		resave: false,
		saveUninitialized: true,
	})
);
app.use(passport.initialize());
app.use(passport.session());

//! Socket Start
let users = [];
const addUser = (userId, socketId) => {
	users = users.filter(user => user.userId !== userId);
	!users.some(user => user.userId === userId) &&
		users.push({ userId, socketId });
};
const removeUser = socketId => {
	users = users.filter(user => user.socketId !== socketId);
};
const getUser = userId => {
	return users.find(user => user.userId === userId);
};
io.on('connection', socket => {
	socket.on('addUser', userId => {
		addUser(userId, socket.id);
		io.emit('getUsers', users);
	});
	socket.on('sendMessage', ({ senderId, receiverId, text }) => {
		const user = getUser(receiverId);
		if (user) {
			io.to(user.socketId).emit('getMessage', {
				senderId,
				text,
			});
		}
	});
	socket.on('disconnect', () => {
		console.log('a user disconnected!');
		removeUser(socket.id);
		io.emit('getUsers', users);
	});
});
//!Socket End

app.use('/users', User);
app.use('/videos', Video);
app.use('/api', Web);
app.use('/payment', Payment);
app.use('/conversation', Conversation);
app.use('/', (req, res, next) => {
	res.send('<h1>Api is working ❤️‍🔥.</h1>');
});
app.all('*', function (req, res, next) {
	next(new ErrorHandler('Bad Request', 404));
});
app.use(errorMiddleware);

server.listen(process.env.PORT, () => {
	console.log(`Running on port ${process.env.PORT} 👍.`);
});

module.exports = app;
