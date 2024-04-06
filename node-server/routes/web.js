var express = require('express');
var router = express.Router();
var passport = require('passport');
var authenticate = require('../middleware/auth');
var webController = require('../controllers/webController');
var videoHandler = require('../middleware/videoHandler');

// ? Web Panel Modules Routes //
router.get('/bid/all', authenticate.verifyUser, webController.getBids);
router.get('/bid/won', authenticate.verifyUser, webController.wonBids);
router.get('/bid/lost', authenticate.verifyUser, webController.lostBids);
router.get('/feed/data', authenticate.verifyUser, webController.getTopCreators);
router.get(
	'/bid/involved',
	authenticate.verifyUser,
	webController.getInvolvedBids
);
router.get(
	'/notifications',
	authenticate.verifyUser,
	webController.getNotifications
);
router.get('/videos', authenticate.verifyUser, webController.searchVideos);
router.post('/bid/create', authenticate.verifyUser, webController.createBid);
router.post('/register', webController.register);
router.post('/login', passport.authenticate('local'), webController.signIn);
router.patch(
	'/bid/update/:id',
	authenticate.verifyUser,
	webController.updateBid
);
router.patch(
	'/notification/setting',
	authenticate.verifyUser,
	webController.changeNotificationSettings
);

//TODO: Job Posting
router.get('/jobs', authenticate.verifyUser, webController.getAllActiveJobs);
router.get('/jobs/past', authenticate.verifyUser, webController.getAllPastJobs);
router.post(
	'/job/create',
	authenticate.verifyUser,
	webController.createJobPosting
);
router.patch(
	'/job/update/:id',
	authenticate.verifyUser,
	webController.stopReceivingRequests
);
router.patch(
	'/job/accept/:id/:sid',
	authenticate.verifyUser,
	webController.acceptRequests
);
router.delete(
	'/job/delete/:id',
	authenticate.verifyUser,
	webController.deleteJob
);

//TODO: Enhancement and filters
router.get(
	'/video/enhance',
	authenticate.verifyUser,
	videoHandler.enhanceVideo
);
router.get('/video/rotate', authenticate.verifyUser, videoHandler.rotateVideo);
router.get(
	'/video/blackwhite',
	authenticate.verifyUser,
	videoHandler.blackAndWhiteVideo
);
router.get(
	'/video/noaudio',
	authenticate.verifyUser,
	videoHandler.removeAudioFromVideo
);

module.exports = router;
