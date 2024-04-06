var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var Notification = new Schema(
	{
		message: { type: String },
		user: { type: mongoose.Types.ObjectId, ref: 'User' },
	},
	{ timestamps: true }
);

module.exports = mongoose.model('Notification', Notification);
