import React from 'react';
import UserOne from '../../assets/img1.png';
import { HOSTNAME } from '../../../Config';
import { useSelector } from 'react-redux';
import { format } from 'timeago.js';

export default function Message({ message, own, currentuser }) {
	const { user } = useSelector(state => state.user);
	var imgVar = '';
	if (own === true) {
		imgVar =
			user.picture === null
				? UserOne
				: `${HOSTNAME}users/picture?path=${user.picture}`;
	} else {
		imgVar =
			currentuser.picture === null
				? UserOne
				: `${HOSTNAME}users/picture?path=${currentuser.picture}`;
	}

	return (
		<div className={`chat ${own ? 'chat-end' : 'chat-start'} p-3`}>
			<div className='chat-image avatar'>
				<div className='w-10 rounded-full'>
					<img src={imgVar} />
				</div>
			</div>
			<div className='chat-header'>
				<time className='text-xs '> {format(message.createdAt)}</time>
			</div>
			<div
				className={`chat-bubble ${
					own ? 'bg-[#EBEBEB] text-black' : 'bg-[#FF8216] text-white'
				}`}>
				{message.text}
			</div>
		</div>
	);
}
