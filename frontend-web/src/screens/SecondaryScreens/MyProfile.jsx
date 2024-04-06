import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';

import MyProfileVideos from '../../Components/NewsFeedComponent/MyProfileVideos';
import UserOne from '../../assets/img1.png';
import { getMyVideosApi, profilePicture } from '../../api/Api';
import { setUser } from '../../redux/userSlice';
import Loader from '../../Components/Loader/Loader';
import { HOSTNAME } from '../../../Config';

import { IoCameraOutline } from 'react-icons/io5';
import Avatar from 'react-avatar-edit';
import { ToastContainer, toast } from 'react-toastify';

const handleModal = async checker => {
	if (checker?.bubbles === true) {
		return document.getElementById('my_modal').showModal();
	}
	document.getElementById('my_modal').close();
};

const MyProfile = () => {
	const [videos, setVideos] = useState([]);
	const [loader, setLoader] = useState(false);
	const [imgCrop, setimgCrop] = useState(false);

	const { user } = useSelector(state => state.user);
	const dispatch = useDispatch();

	const fetchVideos = async () => {
		setLoader(true);
		const res = await getMyVideosApi();
		setVideos(res);
		setLoader(false);
	};

	const onCrop = view => {
		setimgCrop(view);
	};

	const saveImage = async () => {
		const res = await profilePicture(imgCrop);
		let userNew = { ...user, picture: res.picture };
		dispatch(setUser(userNew));
		handleModal(true);
		return toast.success(`Picture updated.`, {
			position: 'bottom-right',
			autoClose: 1500,
			hideProgressBar: false,
			closeOnClick: true,
			pauseOnHover: true,
			draggable: true,
			progress: undefined,
			theme: 'dark',
		});
	};

	useEffect(() => {
		fetchVideos();
	}, []);

	if (loader) {
		return <Loader />;
	}
	return (
		<div className='flex w-full flex-col'>
			<div className=' flex flex-wrap justify-between'>
				<div className='w-full sm:w-[65%]'>
					<div className='relative my-[22px] flex w-full flex-col items-center'>
						<div className=' mb-[22px] w-[85%] rounded-[10px] bg-white shadow-xl'>
							<div className='p-4 '>
								<div className='avatar mb-[20px]  flex w-full flex-row  justify-center'>
									<div className='w-[80px] justify-center  rounded-full border-[3px] border-[#FF8216]'>
										<img
											src={
												user.picture !== null
													? `${HOSTNAME}users/picture?path=${user.picture}`
													: UserOne
											}
										/>
									</div>
									<div
										className='mt-[auto] bg-[#FF8216] rounded-full p-[2px] ml-[-20px] cursor-pointer'
										onClick={handleModal}>
										<IoCameraOutline
											size={20}
											color='white'
										/>
									</div>
								</div>
								<div className='text-center justify-center gap-y-2 mb-[20px]'>
									<p className='text-lg font-bold text-[#FF8216]'>
										{user.username}
									</p>
									<p>{user.email}</p>
								</div>

								<div className='flex flex-row items-center justify-between border-t-[2px] border-t-[#939393] p-4'>
									<div className='cursor-pointer  items-center'>
										<p className='font-bold text-[#939393]'>
											Followers
										</p>
										<p className='text-center  text-lg font-bold text-[#FF8216]'>
											{user.followers?.length || 0}
										</p>
									</div>
									<div className='  cursor-pointer  items-center'>
										<p className='font-bold text-[#939393]'>
											Following
										</p>
										<p className='text-center  text-lg font-bold text-[#FF8216]'>
											{user?.following?.length || 0}
										</p>
									</div>
									<div className='  cursor-pointer  items-center'>
										<p className='font-bold text-[#939393]'>
											Videos
										</p>
										<p className='text-center  text-lg font-bold text-[#FF8216]'>
											{videos.length || 0}
										</p>
									</div>
								</div>
							</div>
						</div>

						<MyProfileVideos data={videos} />
					</div>
				</div>

				{/* Second Column View */}

				<div className='hidden w-full sm:block sm:w-[33%] '></div>

				<dialog id='my_modal' className='modal '>
					<div className='modal-box w-[400px] bg-white'>
						<form method='dialog'>
							{/* if there is a button in form, it will close the modal */}
							<button className='btn btn-circle btn-ghost btn-sm absolute right-2 top-2 bottom-2'>
								✕
							</button>
						</form>
						<div className='flex flex-row items-center justify-between'>
							<Avatar width={200} height={200} onCrop={onCrop} />
							<button
								className='h-[30px] w-[130px] rounded-[10px] bg-[#FF8216] text-[15px] font-bold text-white hover:bg-[#F7A541] active:bg-[#FFD583]'
								style={{ transition: '0.3s' }}
								onClick={saveImage}>
								Save
							</button>
						</div>
					</div>
				</dialog>
			</div>
			<ToastContainer />
		</div>
	);
};

export default MyProfile;
