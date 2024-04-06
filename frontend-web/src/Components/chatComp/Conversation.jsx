import { useEffect, useState } from 'react';
import UserOne from '../../assets/img1.png';
import axiosClient from '../../../axiosClient';
import { HOSTNAME } from '../../../Config';

export default function Conversation({ conversation, currentUser }) {
	const [user, setUser] = useState('');

	// get user engaged in converstion
	useEffect(() => {
		const friendId = conversation.members.find(m => m !== currentUser._id);

		const getUser = async () => {
			try {
				const res = await axiosClient.get('/users?userId=' + friendId);
				setUser(res.data);
			} catch (err) {
				console.log(err);
			}
		};
		getUser();
	}, [currentUser, conversation]);
	return (
		<div className='flex p-2 items-center hover:bg-[#EBEBEB] cursor-pointer'>
			<img
				src={
					user.picture !== null
						? `${HOSTNAME}users/picture?path=${user.picture}`
						: UserOne
				}
				alt='User'
				className='w-[40px] h-[40px] rounded-full mr-[20px]'
			/>
			<span className='font-semibold text-[black]'>{user.username}</span>
		</div>
	);
}
