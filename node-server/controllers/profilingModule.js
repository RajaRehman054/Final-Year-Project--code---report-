require('dotenv').config();
var express = require('express');
var passport = require('passport');
var bodyParser = require('body-parser');
var router = express.Router();
var nodemailer = require('nodemailer');
var path = require('path');
var fs = require('fs');
var mongoose = require('mongoose');
router.use(bodyParser.json());

var authenticate = require('../middleware/auth');
var asyncHandler = require('../middleware/asyncHandler');
var ErrorHandler = require('../utils/error');

var User = require('../models/user');
var Otp = require('../models/otp');
var Posting = require('../models/posting');
var Notification = require('../models/notification');
var Transaction = require('../models/transaction');

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

exports.signIn = asyncHandler(async (req, res) => {
	if (req.user.buyer) {
		return res.status(400).json('UnAuthorized');
	}
	let token = authenticate.getToken({ _id: req.user._id });
	res.status(200).json({
		success: true,
		token: token,
		user: req.user._id,
	});
});

exports.setFCM = asyncHandler(async (req, res) => {
	let update = { fcm: req.body.token };
	await User.findByIdAndUpdate(req.user._id, update);
	res.status(204).json({});
});

exports.getUser = asyncHandler(async (req, res) => {
	const user = await User.findById(req.user.id).populate(
		'followers.user following.user'
	);
	res.status(200).json({ user });
});

exports.getOtherUser = asyncHandler(async (req, res) => {
	const userId = req.query.userId;
	const username = req.query.username;
	const user = userId
		? await User.findById(userId)
		: await User.findOne({ username: username });
	const { password, updatedAt, ...other } = user._doc;
	res.status(200).json(other);
});

exports.getOtp = asyncHandler(async (req, res, next) => {
	var exists = await User.findOne({ email: req.params.email });

	if (!exists) {
		next(new ErrorHandler('Email does not exist', 404));
	} else {
		var existing = await Otp.find({ email: req.params.email });
		if (existing.length > 0) {
			await Otp.deleteOne({ email: req.params.email });
		}
		var a = Math.floor(1000 + Math.random() * 9000).toString();
		var code = a.substring(-2);
		await Otp.create({ token: code, email: req.params.email });
		let transport = nodemailer.createTransport({
			host: 'smtp.gmail.com',
			port: 465,
			secure: true,
			auth: {
				user: process.env.EMAIL,
				pass: process.env.EMAIL_PASSWORD,
			},
		});
		const mailOptions = {
			from: process.env.EMAIL,
			to: req.params.email,
			subject: 'OTP Verification',
			text: `Your four-digit verification code is: ${code}`,
		};

		transport.sendMail(mailOptions, function (err, info) {
			if (err) {
				next(new ErrorHandler('Internal Server Error', 500));
			} else {
				res.status(200).json();
			}
		});
	}
});

exports.verifyOtp = asyncHandler(async (req, res, next) => {
	let otp = req.params.otp;
	let email = req.params.email;
	let doc = await Otp.findOne({ email: email });
	if (doc && otp === doc.token) {
		await Otp.deleteOne({ email: email });
		res.status(200).json();
	} else {
		res.status(404).json({ message: 'Invalid or Expired token' });
	}
});

exports.passwordReset = asyncHandler(async (req, res, next) => {
	let user = await User.findOne({ email: req.body.email });
	let newUser = await user.setPassword(req.body.password);
	newUser.save();
	res.status(204).json();
});

exports.passwordChange = asyncHandler(async (req, res, next) => {
	let user = await User.findById(req.user.id);
	let newUser = await user.setPassword(req.body.new_password);
	newUser.save();
	res.status(204).json();
});

exports.profilePicture = asyncHandler(async (req, res, next) => {
	let update = { picture: req.source };
	await User.findByIdAndUpdate(req.user._id, update);
	res.status(204).json();
});

exports.profilePictureWeb = asyncHandler(async (req, res, next) => {
	if (req.user.picture) {
		fs.unlink(path.resolve(__dirname, `..${req.user.picture}`), err => {
			if (err) {
				console.log(err);
			}
		});
	}
	var data = req.body.picture.replace(/^data:image\/\w+;base64,/, '');
	const binaryData = Buffer.from(data, 'base64');
	const filePath = `files/pictures/${Date.now()}-${Math.round(
		Math.random() * 1e9
	)}.jpg`;
	fs.writeFile(filePath, binaryData, async err => {
		if (err) {
			return res.status(500).send('Internal Server Error');
		} else {
			let update = { picture: `/${filePath}` };
			await User.findByIdAndUpdate(req.user._id, update);
			res.status(200).json({ picture: `/${filePath}` });
		}
	});
});

exports.editProfile = asyncHandler(async (req, res, next) => {
	let exist = await User.findOne({ email: req.body.email });
	if (exist && exist.id !== req.user.id) {
		return res.status(400).send('Unauthorized');
	}
	const user = await User.findByIdAndUpdate(req.user.id, req.body);
	res.status(200).json({ user });
});

exports.getPicture = asyncHandler(async (req, res, next) => {
	res.sendFile(path.join(__dirname, `..${req.query.path}`), function (err) {
		if (err) {
			next(err);
		}
	});
});

exports.updateWalletInc = asyncHandler(async (req, res) => {
	await User.findByIdAndUpdate(req.params.user, {
		$inc: {
			wallet: req.params.amount,
		},
	});
	await Transaction.create({
		amount: req.params.amount,
		user: req.params.user,
	});
	res.redirect(process.env.CLIENT_URL);
});

exports.updateWalletDec = asyncHandler(async (req, res) => {
	await User.findByIdAndUpdate(req.user.id, {
		$inc: {
			wallet: -req.body.amount,
		},
	});
	res.status(204).json();
});

exports.addFollower = asyncHandler(async (req, res, next) => {
	await User.findByIdAndUpdate(req.user.id, {
		$push: { following: { user: req.params.id } },
	});
	await User.findByIdAndUpdate(req.params.id, {
		$push: { followers: { user: req.user.id } },
	});
	res.status(204).json({});
});

exports.removeFollower = asyncHandler(async (req, res, next) => {
	await User.findByIdAndUpdate(req.user.id, {
		$pull: { following: { user: req.params.id } },
	});
	await User.findByIdAndUpdate(req.params.id, {
		$pull: { followers: { user: req.user.id } },
	});
	res.status(204).json({});
});

//TODO: Job Posting
exports.addRequests = asyncHandler(async (req, res) => {
	let posting = await Posting.findById(req.params.id);
	if (posting.requests.length > 0) {
		const userId = new mongoose.Types.ObjectId(req.user.id);
		for (let index = 0; index < posting.requests.length; index++) {
			if (userId.equals(posting.requests[index].user)) {
				return res
					.status(409)
					.json({ message: 'You have already applied for the job' });
			}
		}
	}
	if (req.user.hirer !== null) {
		return res
			.status(400)
			.json({ message: 'You are already hired for a job.' });
	}
	await Posting.findByIdAndUpdate(req.params.id, {
		$push: { requests: { user: req.user.id } },
	});
	let notifier = await User.findById(posting.creator);
	if (notifier.notfication) {
		await Notification.create({
			user: posting.creator,
			message: `You have received the job request for ${posting.details} of ${req.user.username}`,
		});
	}
	res.status(204).json({});
});

exports.getCurrentJobs = asyncHandler(async (req, res) => {
	let postings = await Posting.find({
		ended: false,
		receiving: true,
		'requests.user': {
			$nin: [new mongoose.Types.ObjectId(req.user.id)],
		},
	});
	res.status(200).json(postings);
});

exports.getAppliedJobs = asyncHandler(async (req, res) => {
	let postings = await Posting.find({
		ended: false,
		receiving: true,
		'requests.user': new mongoose.Types.ObjectId(req.user.id),
	});
	res.status(200).json(postings);
});

exports.getCurrentHiredJob = asyncHandler(async (req, res) => {
	let postings = await Posting.find({
		ended: false,
		receiving: true,
		requests: {
			$elemMatch: {
				user: new mongoose.Types.ObjectId(req.user.id),
				accept: true,
			},
		},
	});
	res.status(200).json(postings);
});
