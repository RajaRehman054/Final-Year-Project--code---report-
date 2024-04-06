var mongoose = require('mongoose');
var passportLocalMongoose = require('passport-local-mongoose');
var Schema = mongoose.Schema;

var User = new Schema({
	name: {
		type: String,
		default: '',
	},
	email: {
		type: String,
		default: '',
	},
	address: {
		type: String,
		default: '',
	},
	phone: {
		type: String,
		default: '',
	},
	username: {
		type: String,
		default: '',
	},
	email: {
		type: String,
		default: '',
	},
	buyer: {
		type: Boolean,
		default: false,
	},
	fcm: {
		type: String,
		default: '',
	},
	picture: { type: String, default: null },
	sales: { type: Number, default: 0 },
	followers: {
		type: [
			{
				user: {
					type: mongoose.Types.ObjectId,
					ref: 'User',
				},
			},
		],
		default: [],
	},
	following: {
		type: [
			{
				user: {
					type: mongoose.Types.ObjectId,
					ref: 'User',
				},
			},
		],
		default: [],
	},
	wallet: { type: Number, default: 0 },
	videos_created: { type: Number, default: 0 },
	bought: { type: Number, default: 0 },
	hirer: { type: mongoose.Types.ObjectId, ref: 'User', default: null },
	notification: { type: Boolean, default: true },
	notification_bid: { type: Boolean, default: true },
	notification_payment: { type: Boolean, default: true },
});
User.plugin(passportLocalMongoose);

module.exports = mongoose.model('User', User);
