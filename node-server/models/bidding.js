var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var bidSchema = new Schema(
	{
		current_highest: {
			type: mongoose.Types.ObjectId,
			ref: 'User',
		},
		current_amount: { type: Number, default: 0 },
		video: {
			type: mongoose.Types.ObjectId,
			ref: 'Video',
		},
		list: {
			type: [
				{
					user: { type: mongoose.Types.ObjectId, ref: 'User' },
					amount: { type: Number },
					paid: { type: Boolean, default: false },
				},
			],
		},
		original_owner: { type: mongoose.Types.ObjectId, ref: 'User' },
		expires: { type: Number },
		expired: { type: Boolean, default: false },
	},
	{ timestamps: true }
);

const Bid = mongoose.model('Bid', bidSchema);
module.exports = Bid;
