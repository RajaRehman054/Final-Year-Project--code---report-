var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var Video = new Schema({
	likes: {
		type: Number,
		default: 0,
	},
	owner: {
		type: mongoose.Types.ObjectId,
		ref: 'User',
	},
	path: {
		type: String,
		default: '',
	},
	thumbnail: {
		type: String,
		default: '',
	},
	bought: {
		type: Boolean,
		default: false,
	},
	created_on: {
		type: Date,
		default: Date.now(),
	},
	description: {
		type: String,
		default: '',
	},
	tags: { type: String, default: '' },
	checked: { type: Boolean, default: false },
	checking: { type: Boolean, default: false },
});

module.exports = mongoose.model('Video', Video);
