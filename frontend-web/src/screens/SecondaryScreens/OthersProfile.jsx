import { useState, useEffect } from 'react';
import OtherProfileVideos from '../../Components/NewsFeedComponent/OtherProfileVideos';
import { useLocation } from 'react-router-dom';
import { HOSTNAME } from '../../../Config';
import {
	getOtherVideosApi,
	addFollower,
	removeFollower,
	getUser,
} from '../../api/Api';
import Loader from '../../Components/Loader/Loader';
import UserOne from '../../assets/img1.png';
import { useSelector } from 'react-redux';
import { useDispatch } from 'react-redux';
import { setUser } from '../../redux/userSlice';

const OthersProfile = () => {
	const [videos, setVideos] = useState([]);
	const [loading, setLoading] = useState(false);
	const [follower, setFollower] = useState(false);

	const { state } = useLocation();

	const { user } = useSelector(state => state.user);
	const dispatch = useDispatch();

	const fetchUsers = async () => {
		const data = await getUser();
		dispatch(setUser(data.user));
	};

	const handleFollow = async () => {
		if (!follower) {
			await addFollower(state._id);
			await fetchUsers();
			setFollower(true);
		} else {
			await removeFollower(state._id);
			await fetchUsers();
			setFollower(false);
		}
	};

	const fetchVideos = async () => {
		setLoading(true);
		const data = await getOtherVideosApi(state._id);
		if (user.following.length === 0) {
			setFollower(false);
		} else {
			const check = user.following.filter(
				elem => elem.user._id === state._id
			);
			if (check.length > 0) {
				setFollower(true);
			}
		}
		setVideos(data);
		setLoading(false);
	};

	useEffect(() => {
		fetchVideos();
	}, []);

	if (loading) {
		return <Loader />;
	}

	return (
		<div className='flex w-full flex-col'>
			<div className=' flex flex-wrap justify-between'>
				{/*First Column View */}
				<div className='w-full sm:w-[65%]'>
					<div className='relative my-[22px] flex w-full flex-col items-center'>
						<div className=' mb-[22px] w-[85%] rounded-[10px] bg-white shadow-xl'>
							<div className='p-4 '>
								<div className='avatar mb-[10px]  flex w-full flex-col items-center justify-center gap-y-2'>
									<div className='w-[80px] justify-center  rounded-full border-[3px] border-[#FF8216]'>
										<img
											src={
												state.picture !== null
													? `${HOSTNAME}users/picture?path=${state.picture}`
													: UserOne
											}
										/>
									</div>
									<p className='text-lg font-bold text-[#FF8216]'>
										{state.username}
									</p>
									<p>{state.email}</p>
								</div>

								<div className='flex flex-row items-center justify-between border-t-[2px] border-t-[#939393] p-4'>
									<div className='  cursor-pointer  items-center'>
										<p className='font-bold text-[#939393]'>
											Followers
										</p>
										<p className='text-center  text-lg font-bold text-[#FF8216]'>
											{state.followers
												? state.followers.length
												: 0}
										</p>
									</div>
									<div className='  cursor-pointer  items-center'>
										<p className='font-bold text-[#939393]'>
											Following
										</p>
										<p className='text-center  text-lg font-bold text-[#FF8216]'>
											{state.following
												? state.following.length
												: 0}
										</p>
									</div>
									<div className='  cursor-pointer  items-center'>
										<p className='font-bold text-[#939393]'>
											Videos
										</p>
										<p className='text-center  text-lg font-bold text-[#FF8216]'>
											{state.buyer
												? state.bought
												: state.created_videos -
												  state.sales}
										</p>
									</div>
								</div>

								<div className='flex flex-row items-center  justify-center p-4'>
									<div className='items-center '>
										<button
											className='mx-[10px] h-[22px] w-[100px] justify-center rounded-[10px] text-[15px]  text-white hover:bg-[#F7A541] active:bg-[#FFD583] sm:h-[35px] sm:w-[200px]'
											style={{
												backgroundColor: follower
													? 'red'
													: '#FF8216',
												transition: '0.3s',
											}}
											onClick={handleFollow}>
											{follower ? (
												<p>Following</p>
											) : (
												<p>Follow</p>
											)}
										</button>
									</div>
								</div>
							</div>
						</div>

						{/*IMported Component */}
						<OtherProfileVideos data={videos} />
					</div>
				</div>

				{/* Second Column View */}

				<div className='hidden w-full sm:block sm:w-[33%] '></div>
			</div>
		</div>
	);
};

export default OthersProfile;
