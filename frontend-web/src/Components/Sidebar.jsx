import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useRef } from 'react';
import Pic from '.././assets/orange-logo.png';
import {
	IoChatbubbleOutline,
	IoWalletOutline,
	IoSettingsOutline,
	IoLogOutOutline,
	IoCashOutline,
	IoPeopleOutline,
	IoArrowBack,
	IoSearchOutline,
} from 'react-icons/io5';
import { MdOutlineDashboard } from 'react-icons/md';
import { clearUser } from '../redux/userSlice';
import { useDispatch } from 'react-redux';

// eslint-disable-next-line react/prop-types
const Sidebar = ({ sidebarOpen, setSidebarOpen }) => {
	const location = useLocation();
	const dispatch = useDispatch();
	const navigate = useNavigate();
	const { pathname } = location;

	const trigger = useRef(null);
	const sidebar = useRef(null);

	// close on click outside
	useEffect(() => {
		const clickHandler = ({ target }) => {
			if (!sidebar.current || !trigger.current) return;
			if (
				!sidebarOpen ||
				sidebar.current.contains(target) ||
				trigger.current.contains(target)
			)
				return;
			setSidebarOpen(false);
		};
		document.addEventListener('click', clickHandler);
		return () => document.removeEventListener('click', clickHandler);
	});

	// close if the esc key is pressed
	useEffect(() => {
		const keyHandler = ({ keyCode }) => {
			if (!sidebarOpen || keyCode !== 27) return;
			setSidebarOpen(false);
		};
		document.addEventListener('keydown', keyHandler);
		return () => document.removeEventListener('keydown', keyHandler);
	});

	const handleSignOut = async () => {
		dispatch(clearUser());
		localStorage.removeItem('jwt');
		window.location.replace('/login');
	};

	return (
		<aside
			ref={sidebar}
			className={` absolute left-0 top-0 z-[40000] flex h-screen w-[255px] flex-col overflow-y-hidden bg-sidebar duration-300 ease-linear lg:static lg:translate-x-0 ${
				sidebarOpen ? 'translate-x-0' : '-translate-x-full'
			}`}>
			<div className='py-5.5 lg:py-6.5 mt-[25px] flex items-center justify-between gap-2 px-6 lg:justify-center'>
				<NavLink to='/'>
					<img
						src={Pic}
						alt='My Image'
						className='h-[50px] w-[120px]'
					/>
				</NavLink>
				<button
					ref={trigger}
					onClick={() => setSidebarOpen(!sidebarOpen)}
					aria-controls='sidebar'
					aria-expanded={sidebarOpen}
					className='block lg:hidden'>
					<IoArrowBack size={26} color='black' />
				</button>
			</div>

			<div className='no-scrollbar flex flex-col overflow-y-auto duration-300 ease-linear'>
				{/* <!-- Sidebar Menu --> */}
				<nav className='mt-5 px-4 py-4 lg:mt-9 lg:px-6'>
					{/* <!-- Menu Group --> */}
					<div>
						<ul className='mb-6 flex flex-col gap-3'>
							{/* <!-- Menu Item Dashboard --> */}
							<li>
								<NavLink
									to='/'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium  duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname === '/' ||
										pathname.includes('newsfeed')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<MdOutlineDashboard
										size={24}
										color='currentColor'
									/>
									News Feed
								</NavLink>
							</li>
							{/* <!-- Menu Item Dashboard --> */}

							{/* <!-- Menu Item Search --> */}
							<li>
								<NavLink
									to='/search'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname.includes('search')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<IoSearchOutline
										size={24}
										color='currentColor'
									/>
									Search
								</NavLink>
							</li>
							{/* <!-- Menu Item Search --> */}

							{/* <!-- Menu Item Calendar --> */}
							<li>
								<NavLink
									to='/JobPost'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname.includes('JobPost')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<IoPeopleOutline
										size={24}
										color='currentColor'
									/>
									Job Post
								</NavLink>
							</li>
							{/* <!-- Menu Item Calendar --> */}

							{/* <!-- Menu Item Profile --> */}
							<li>
								<NavLink
									to='/bids'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname.includes('bids')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<IoCashOutline
										size={24}
										color='currentColor'
									/>
									Bids
								</NavLink>
							</li>
							{/* <!-- Menu Item Profile --> */}

							{/* <!-- Menu Item Tables --> */}
							<li>
								<NavLink
									to='/chat'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname.includes('chat')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<IoChatbubbleOutline
										size={24}
										color='currentColor'
									/>
									Chat
								</NavLink>
							</li>
							{/* <!-- Menu Item Tables --> */}

							{/* <!-- Menu Item Settings --> */}
							<li>
								<NavLink
									to='/payment'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname.includes('payment')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<IoWalletOutline
										size={24}
										color='currentColor'
									/>
									Payment
								</NavLink>
							</li>
							{/* <!-- Menu Item Settings --> */}

							{/* <!-- Menu Item Settings --> */}
							<li>
								<NavLink
									to='/settings'
									className={`group relative flex items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216] ${
										pathname.includes('settings')
											? 'bg-hover text-[#FF8216]'
											: 'text-text'
									}`}>
									<IoSettingsOutline
										size={24}
										color='currentColor'
									/>
									Settings
								</NavLink>
							</li>
							{/* <!-- Menu Item Settings --> */}
						</ul>
					</div>
				</nav>
			</div>

			<div className='absolute bottom-0 w-full px-6'>
				<ul className='mb-6 flex flex-col gap-1.5'>
					{/* <!-- Menu Item Chart --> */}
					<li
						className={`group relative flex cursor-pointer items-center gap-2.5 rounded-[10px] px-4 py-2 text-[17px] font-medium text-text duration-300 ease-in-out hover:bg-hover hover:text-[#FF8216]`}
						onClick={handleSignOut}>
						<IoLogOutOutline
							size={24}
							color='currentColor'
							onClick={handleSignOut}
						/>
						Sign Out
					</li>
					{/* <!-- Menu Item Chart --> */}
				</ul>
			</div>
		</aside>
	);
};

export default Sidebar;
