import React, { useEffect, useState, useRef } from 'react';
import { IoInformationCircleOutline, IoPaperPlane } from 'react-icons/io5';
import { BiMessageSquareEdit } from 'react-icons/bi';
import UserOne from '../../assets/img1.png';
import Message from '../../Components/chatComp/Message';
import Conversation from '../../Components/chatComp/Conversation';
import Loader from '../../Components/Loader/Loader';
import axiosClient from '../../../axiosClient';
import { useSelector } from 'react-redux';
import { io } from 'socket.io-client';
import { HOSTNAME } from '../../../Config';
import { useNavigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';

const handleModal = async checker => {
	if (checker?.bubbles === true) {
		return document.getElementById('my_modal').showModal();
	}
	document.getElementById('my_modal').close();
};

const Chat = () => {
	const { user } = useSelector(state => state.user);
	const [conversations, setConversations] = useState([]);
	const [currentChat, setCurrentChat] = useState(null);
	const [messages, setMessages] = useState([]);
	const [newMessage, setNewMessage] = useState('');
	const [arrivalMessage, setArrivalMessage] = useState(null);
	const [currentChatUser, setCurrentChatUser] = useState('');
	const newChatArray = Array.isArray(user.following) ? user.following : [];
	const [loading, setLoading] = useState(false);
	const navigate = useNavigate();
	const socket = useRef();
	const scrollRef = useRef();

	useEffect(() => {
		socket.current = io('ws://localhost:4000');
		socket.current.on('getMessage', data => {
			setArrivalMessage({
				sender: data.senderId,
				text: data.text,
				createdAt: Date.now(),
			});
		});
	}, []);

	useEffect(() => {
		arrivalMessage &&
			currentChat?.members.includes(arrivalMessage.sender) &&
			setMessages(prev => [...prev, arrivalMessage]);
	}, [arrivalMessage, currentChat]);

	useEffect(() => {
		socket.current.emit('addUser', user._id);
	}, [user]);

	//get user of current chat
	const getUser = async c => {
		setCurrentChat(c);
		var datamember = c.members.find(m => m !== user._id);
		try {
			const res = await axiosClient.get('/users?userId=' + datamember);
			setCurrentChatUser(res.data);
		} catch (err) {
			console.log(err);
		}
	};

	// get converstions of the current user
	useEffect(() => {
		const getConversations = async () => {
			try {
				const res = await axiosClient.get('/conversation/' + user._id);
				setConversations(res.data);
			} catch (err) {
				console.log(err);
			}
		};
		getConversations();
	}, [user._id]);

	// get messages of a current converstion
	useEffect(() => {
		const getMessages = async () => {
			try {
				const res = await axiosClient.get(
					'/conversation/getmessages/' + currentChat?._id
				);
				getConversationNow();
				setMessages(res.data);
			} catch (err) {
				console.log(err);
			}
		};
		getMessages();
	}, [currentChat]);

	//Create new Converstion from following
	const createConversation = async e => {
		const convoArr = [];
		var exists = false;
		for (let i = 0; i < conversations.length; i++) {
			convoArr[i] = conversations[i].members.find(m => m !== user._id);
		}
		for (let i = 0; i < conversations.length; i++) {
			if (convoArr[i] === e.user._id) {
				exists = true;
				break;
			}
		}

		if (exists === true) {
			handleModal();
			return toast.warn('Coversation already exists with user', {
				position: 'bottom-center',
				autoClose: 5000,
				hideProgressBar: false,
				closeOnClick: true,
				pauseOnHover: true,
				draggable: true,
				progress: undefined,
				theme: 'dark',
			});
		} else {
			const convo = {
				senderId: user._id,
				receiverId: e.user._id,
			};
			try {
				await axiosClient.post('/conversation', convo);
				setLoading(true);
				await getConversationNow();
				handleModal();
			} catch (err) {
				console.log(err);
			} finally {
				setLoading(false);
			}
		}
	};

	const getConversationNow = async () => {
		try {
			const res = await axiosClient.get('/conversation/' + user._id);
			setConversations(res.data);
		} catch (err) {
			console.log(err);
		}
	};

	// create new message
	const handleSubmit = async e => {
		e.preventDefault();
		const message = {
			sender: user._id,
			text: newMessage,
			conversationId: currentChat._id,
		};

		const receiverId = currentChat.members.find(
			member => member !== user._id
		);

		socket.current.emit('sendMessage', {
			senderId: user._id,
			receiverId,
			text: newMessage,
		});

		try {
			const res = await axiosClient.post(
				'/conversation/newmessage',
				message
			);
			setMessages([...messages, res.data]);
			setNewMessage('');
		} catch (err) {
			console.log(err);
		}
	};

	useEffect(() => {
		scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages]);

	return (
		<div
			className='flex w-full flex-col h-screen'
			style={{ height: 'calc(100vh - 95px)' }}>
			{/*Top Heading */}
			<div className='flex w-full '>
				<h1 className='text-[20px] font-bold'>Chat</h1>
			</div>

			{/*View Area divided in two */}
			<div className='flex flex-wrap flex-1 overflow-y-auto justify-between rounded-xl bg-white p-3  shadow-xl my-[22px]'>
				{/* First Column View */}
				<div className='hidden w-full sm:block sm:w-[30%] border-r-[2px] border-gray'>
					<div className='flex border-b-[2px] border-b-gray items-center'>
						<h1 className='text-[16px] font-bold p-3 ml-auto text-[black] '>
							{user.username}
						</h1>
						<span
							className='tooltip tooltip-left cursor-pointer ml-auto mr-3'
							data-tip='Create new message'
							onClick={handleModal}>
							<BiMessageSquareEdit size={24} color='black' />
						</span>
					</div>
					{loading ? (
						<Loader />
					) : conversations.length > 0 ? (
						conversations.map(c => (
							<div key={c._id} onClick={() => getUser(c)}>
								<Conversation
									key={c._id}
									conversation={c}
									currentUser={user}
								/>
							</div>
						))
					) : (
						<div className='flex p-2 items-center '>
							<p className='font-bold text-[black] '>
								No Conversations yet
							</p>
						</div>
					)}
				</div>

				{/*Second Column */}
				<div className=' relative flex flex-col w-full sm:w-[70%] '>
					{currentChat ? (
						<>
							{/*Upper header */}
							<div className='flex border-b-[2px] border-b-gray items-center px-3'>
								<img
									src={
										currentChatUser.picture === null
											? UserOne
											: `${HOSTNAME}users/picture?path=${currentChatUser.picture}`
									}
									alt='User'
									className='w-[40px] h-[40px] rounded-full mr-[20px]'
								/>
								<span>
									<h1 className='text-[16px] text-[black] font-bold  p-3 '>
										{currentChatUser.username}
									</h1>
								</span>
								<span
									className='tooltip tooltip-left ml-auto cursor-pointer'
									data-tip='Visit Profile'
									onClick={() =>
										navigate(`/otherProfile`, {
											state: currentChatUser,
										})
									}>
									<IoInformationCircleOutline
										size={24}
										color='black'
									/>
								</span>
							</div>
							{/*chat box area */}
							<div className='max-h-[320px] overflow-y-auto'>
								{messages.map((m, index) => (
									<div key={index} ref={scrollRef}>
										<Message
											key={index}
											message={m}
											own={m.sender === user._id}
											currentuser={currentChatUser}
										/>
									</div>
								))}
							</div>
							{/*chat box input area */}
							<div className='flex w-full items-center p-3 justify-between absolute bottom-0'>
								<label
									htmlFor='searchInput'
									className='h-[50px] w-full mr-[10px] flex-row items-center justify-center gap-x-6 rounded-[50px] bg-[#F5F5F5] px-3 text-[#AEB6CF] focus:outline-none lg:flex lg:justify-start'
									style={{
										boxShadow:
											'0px 4px 10px 0px rgba(0, 0, 0, 0.10)',
										cursor: 'text',
									}}>
									<input
										type='text'
										id='searchInput' // Add an id to the input
										className='w-full p-2 bg-[#F5F5F5] text-[#868DA3] placeholder:text-[#AEB6CF] focus:outline-none'
										placeholder='Type your message here....'
										onChange={e =>
											setNewMessage(e.target.value)
										}
										value={newMessage}
									/>
								</label>
								<span>
									<div
										className='flex rounded-full h-[50px] w-[50px] bg-[#FF8216] items-center justify-center cursor-pointer'
										onClick={handleSubmit}>
										<IoPaperPlane size={24} color='white' />
									</div>
								</span>
							</div>
						</>
					) : (
						<span className='m-[auto] text-[20px] font-semibold'>
							Open a conversation to start a chat.
						</span>
					)}
				</div>
			</div>
			{/*create new chat Modal */}

			<dialog id='my_modal' className='modal '>
				<div className='modal-box w-[400px] bg-white'>
					<form method='dialog'>
						<button
							className='btn btn-circle btn-ghost btn-sm absolute right-2 top-2'
							onClick={handleModal}>
							✕
						</button>
					</form>
					<div>
						<h3 className='text-center text-lg font-bold text-[#FF8216]'>
							Start Chat with Followings
						</h3>
					</div>
					{newChatArray.length !== 0 ? (
						newChatArray.map((element, index) => (
							<div
								key={index}
								className=' mb-[10px]  flex h-[80px] cursor-pointer flex-row items-center justify-between  rounded-[20px] bg-[white]  p-4  hover:bg-[#EBEBEB]'
								onClick={() => createConversation(element)}>
								<div className='avatar items-center'>
									<div className='w-[50px] rounded-full'>
										<img
											src={
												element.user.picture === null
													? UserOne
													: `${HOSTNAME}users/picture?path=${element.user.picture}`
											}
										/>
									</div>
								</div>
								<div className='mr-[auto] px-5'>
									<p className='font-bold text-[black]'>
										{element.user.username}
									</p>
								</div>
							</div>
						))
					) : (
						<div className='flex h-[80px] flex-row items-center justify-between  rounded-[20px] bg-[white]  p-4'>
							<p className='font-bold text-[black] text-lg'>
								No Followers Yet
							</p>
						</div>
					)}
				</div>
			</dialog>
			<ToastContainer />
		</div>
	);
};

export default Chat;
