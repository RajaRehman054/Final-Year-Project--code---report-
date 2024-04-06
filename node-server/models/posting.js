var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var postingSchema = new Schema(
	{
		creator: { type: mongoose.Types.ObjectId, ref: 'User' },
		price: { type: Number, default: 0 },
		expiry_date: { type: String, default: '' },
		time: { type: Number },
		details: { type: String, default: '' },
		receiving: { type: Boolean, default: true },
		requests: {
			type: [
				{
					user: {
						type: mongoose.Types.ObjectId,
						ref: 'User',
					},
					accept: {
						type: Boolean,
						default: false,
					},
					action: { type: Boolean, default: false },
				},
			],
			default: [],
		},
		ended: { type: Boolean, default: false },
	},
	{ timestamps: true }
);

const Posting = mongoose.model('Posting', postingSchema);
module.exports = Posting;
