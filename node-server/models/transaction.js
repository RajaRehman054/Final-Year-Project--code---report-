var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var Transaction = new Schema(
	{
		amount: { type: Number, required: true },
		user: { type: mongoose.Types.ObjectId, ref: 'User' },
	},
	{ timestamps: true }
);

module.exports = mongoose.model('Transaction', Transaction);
