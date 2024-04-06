var express = require('express');
var bodyParser = require('body-parser');
var passport = require('passport');
var router = express.Router();
router.use(bodyParser.json());

var asyncHandler = require('../middleware/asyncHandler');
var ErrorHandler = require('../utils/error');
var authenticate = require('../middleware/auth');

var Video = require('../models/video');
var Bid = require('../models/bidding');
var User = require('../models/user');
var Posting = require('../models/posting');
var Conversation = require('../models/conversation');
var Notification = require('../models/notification');

var pushNotification = require('../utils/pushNotifications');

exports.register = async (req, res, next) => {
	var exists = await User.findOne({ email: req.body.email });
	if (exists) {
		next(new ErrorHandler('Email already associated with an account', 409));
	} else {
		try {
			const user = await User.register(
				new User({
					username: req.body.username,
					email: req.body.email,
					buyer: true,
				}),
				req.body.password
			);
			if (user) {
				try {
					await user.save();
					passport.authenticate('local')(req, res, () => {
						res.status(201).json({
							success: true,
							status: 'Registration Successful!',
						});
					});
				} catch (error) {
					return next(error);
				}
			}
		} catch (error) {
			return next(new ErrorHandler('Username Already Exists.', 409));
		}
	}
};

exports.signIn = asyncHandler(async (req, res, next) => {
	if (!req.user.buyer) {
		return res.status(400).send('Unauthorized');
	}
	let token = authenticate.getToken({ _id: req.user._id });
	res.status(200).json({
		success: true,
		token: token,
		user: req.user._id,
	});
});

exports.getTopCreators = asyncHandler(async (req, res) => {
	const users = await User.find({ buyer: false }).sort({
		created_videos: -1,
	});

	const channels = await User.find({ buyer: true }).sort({ bought: -1 });

	res.status(200).json({ users, channels });
});

//? BIDDING
exports.createBid = asyncHandler(async (req, res) => {
	let owner = await Video.findById(req.body.video).select('owner');
	let { amount, video } = req.body;
	const newBid = {
		current_amount: amount,
		current_highest: req.user.id,
		video,
		list: [{ user: req.user.id, amount }],
		expires: Date.now() + 7200000,
		original_owner: owner.owner,
	};
	await Video.findByIdAndUpdate(req.body.video, {
		bought: true,
	});
	await User.findByIdAndUpdate(req.user._id, {
		$inc: { wallet: -req.body.amount },
	});
	let data = await User.findById(owner.owner);
	data.fcm !== ''
		? await pushNotification.startedBidNotification([data.fcm])
		: null;
	await Notification.create({
		user: req.user._id,
		message: `You have started a bidding war on video of ${data.username}`,
	});
	await Bid.create(newBid);
	res.status(201).json({ message: 'Bid created' });
});

exports.updateBid = asyncHandler(async (req, res) => {
	let check = true;
	let bid = await Bid.findById(req.params.id);
	let data = await User.findById(bid.original_owner);
	if (req.body.amount <= bid.current_amount) {
		return res.status(400).json({ success: false });
	}
	for (let i = 0; i < bid.list.length; i++) {
		if (
			bid.list[i].user.equals(req.user._id) &&
			bid.list[i].paid === false
		) {
			if (req.user.wallet + bid.list[i].amount < req.body.amount) {
				return res.status(400).json({ success2: false });
			}
		}
		check = false;
	}
	if (!check) {
		if (req.user.wallet < req.body.amount) {
			return res.status(400).json({ success2: false });
		}
	}
	await Notification.create({
		user: bid.current_highest,
		message: `You have been outbid on a video of ${data.username}`,
	});
	for (let index = 0; index < bid.list.length; index++) {
		if (!bid.list[index].paid) {
			await User.findByIdAndUpdate(bid.list[index].user, {
				$inc: { wallet: bid.list[index].amount },
			});
			await Bid.updateOne(
				{ _id: req.params.id, 'list._id': bid.list[index]._id },
				{ $set: { 'list.$.paid': true } }
			);
		}
	}
	await User.findByIdAndUpdate(req.user._id, {
		$inc: { wallet: -req.body.amount },
	});
	await Bid.findByIdAndUpdate(req.params.id, {
		current_amount: req.body.amount,
		current_highest: req.user.id,
		$push: { list: { user: req.user.id, amount: req.body.amount } },
	});
	res.status(201).json({ message: 'bid added' });
});

exports.getInvolvedBids = asyncHandler(async (req, res) => {
	let data = [];
	const bids = await Bid.find({ 'list.user': { $in: req.user.id } })
		.populate('video')
		.sort({
			created_at: -1,
		})
		.populate('list.user');
	var obj = {};
	bids.forEach(element => {
		let obj1 = {
			video: element.video,
			expires: element.expires,
			current_highest: element.current_amount,
		};
		let checked = false;
		element.list.forEach(elem => {
			if (elem.user.id === req.user.id) {
				if (checked) {
					data.push(obj);
				}
				obj = {
					...obj1,
					picture: elem.user.picture,
					amount: elem.amount,
				};
				checked = true;
			}
		});
		data.push(obj);
	});
	res.status(200).json({ data });
});

exports.wonBids = asyncHandler(async (req, res) => {
	const bids = await Bid.find({
		expired: true,
		current_highest: req.user.id,
	}).populate('video');
	res.status(200).json(bids);
});

exports.lostBids = asyncHandler(async (req, res) => {
	const bids = await Bid.find({
		'list.user': { $in: req.user.id },
		expired: true,
		current_highest: { $ne: req.user.id },
	}).populate('video');
	res.status(200).json(bids);
});

exports.getBids = asyncHandler(async (req, res) => {
	const bids = await Bid.find({ expired: false }).populate(
		'video current_highest list.user original_owner'
	);
	res.status(200).json(bids);
});

//? JOB POSTINGS
exports.createJobPosting = asyncHandler(async (req, res) => {
	var date = new Date(req.body.expiry_date);
	var time = date.getTime();
	await Posting.create({ ...req.body, time, creator: req.user.id });
	res.status(201).json({ message: 'Job Posting created successfully.' });
});

exports.getAllActiveJobs = asyncHandler(async (req, res) => {
	var jobs = await Posting.find({
		creator: req.user.id,
		ended: false,
	}).populate('creator requests.user');
	res.status(200).json({ jobs });
});

exports.getAllPastJobs = asyncHandler(async (req, res) => {
	var jobs = await Posting.find({
		creator: req.user.id,
		ended: true,
	}).populate('creator requests.user');
	res.status(200).json({ jobs });
});

exports.stopReceivingRequests = asyncHandler(async (req, res) => {
	const post = await Posting.findById(req.params.id);
	await Posting.findByIdAndUpdate(req.params.id, {
		receiving: post.receiving ? false : true,
	});
	res.status(204).json({});
});

exports.acceptRequests = async (req, res) => {
	let requestor = await User.findById(req.params.sid);
	let posting = await Posting.findById(req.params.id);
	if (req.query.reject === 'true') {
		await Posting.updateOne(
			{ _id: req.params.id, 'requests.user': req.params.sid },
			{ $set: { 'requests.$.accept': false, 'requests.$.action': true } }
		);
		requestor.fcm !== ''
			? await pushNotification.acceptedJobRequest(true, posting.details, [
					requestor.fcm,
			  ])
			: null;
		return res.status(204).json({});
	}
	if (posting.price > req.user.wallet) {
		return res.status(400).json({ message: 'Not enough balance.' });
	}
	await Posting.updateOne(
		{ _id: req.params.id, 'requests.user': req.params.sid },
		{ $set: { 'requests.$.accept': true, 'requests.$.action': true } }
	);
	requestor.fcm !== ''
		? await pushNotification.acceptedJobRequest(false, posting.details, [
				requestor.fcm,
		  ])
		: null;
	await Posting.updateMany(
		{
			requests: {
				$elemMatch: {
					user: req.params.sid,
					accept: false,
				},
			},
		},
		{
			$pull: {
				requests: {
					user: req.params.sid,
					accept: false,
				},
			},
		}
	);
	await User.findByIdAndUpdate(req.params.sid, {
		hirer: req.user.id,
		$inc: {
			wallet: posting.price,
		},
	});
	await User.findByIdAndUpdate(req.user.id, {
		$inc: {
			wallet: -posting.price,
		},
	});
	await Notification.create({
		user: req.user._id,
		message: `You have accepted the job request for ${posting.details} of ${requestor.username}`,
	});
	let prevConv = await Conversation.findOne({
		members: { $all: [req.user.id, req.params.sid] },
	});
	if (!prevConv) {
		const newConversation = new Conversation({
			members: [req.user.id, req.params.sid],
		});
		await newConversation.save();
	}
	res.status(204).json({});
};

exports.deleteJob = asyncHandler(async (req, res) => {
	let posting = await Posting.findById(req.params.id);
	if (posting.requests.length > 0) {
		for (let index = 0; index < posting.requests.length; index++) {
			if (posting.requests[index].accept === true) {
				return res.status(400).json({
					message:
						'Job cannot be deleted as there is an accepted request.',
				});
			}
		}
	}
	await Posting.findByIdAndDelete(req.params.id);
	res.status(200).json({ message: 'Job deleted successfully.' });
});

//? Notifications
exports.getNotifications = asyncHandler(async (req, res) => {
	var notifications = await Notification.find({ user: req.user._id }).sort({
		createdAt: -1,
	});
	res.status(200).json(notifications);
});

exports.changeNotificationSettings = asyncHandler(async (req, res) => {
	await User.findByIdAndUpdate(req.user.id, {
		notification: req.body.notification,
		notification_bid: req.body.notification_bid,
		notification_payment: req.body.notification_payment,
	});
	res.status(204).json({});
});

//?Search
exports.searchVideos = asyncHandler(async (req, res) => {
	if (req.query.type === 'username') {
		var users = await User.find({
			username: { $regex: new RegExp(req.query.value, 'i') },
		}).select('_id');
		const videos = await Video.find({
			owner: { $in: users },
			bought: false,
			checked: true,
		}).populate('owner');
		return res.status(200).json(videos);
	}
	var videos = await Video.find({
		[req.query.type]: {
			$regex: new RegExp(req.query.value, 'i'),
		},
		bought: false,
		checked: true,
	}).populate('owner');
	res.status(200).json(videos);
});
