const mongoose = require('mongoose');
var Schema = mongoose.Schema;

const ConversationSchema = new Schema(
	{
		members: {
			type: Array,
		},
	},
	{ timestamps: true }
);

module.exports = mongoose.model('Conversation', ConversationSchema);
