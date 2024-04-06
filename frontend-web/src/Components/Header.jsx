import DropdownNotification from './DropdownNotification';
import { Link } from 'react-router-dom';
import UserOne from '../assets/img1.png';

import { IoWallet } from 'react-icons/io5';
import { useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';

import { HOSTNAME } from '../../Config';

const Header = ({ sidebarOpen, setSidebarOpen, data }) => {
	const { user } = useSelector(state => state.user);
	const navigate = useNavigate();

	return (
		<header className='drop-shadow-1 sticky top-0 z-[30000] flex w-full bg-white'>
			<div className='shadow-2 flex flex-grow items-center justify-between px-4 py-4 md:px-6 2xl:px-11'>
				<div className='flex items-center gap-2 sm:gap-4 lg:hidden'>
					<button
						aria-controls='sidebar'
						onClick={e => {
							e.stopPropagation();
							setSidebarOpen(!sidebarOpen);
						}}
						className='z-99999 border-stroke block rounded-sm border border-gray-300 bg-white p-1.5 shadow-sm lg:hidden'>
						<span className='relative block h-[22px] w-[22px] cursor-pointer'>
							<span className='absolute right-0 block h-full w-full'>
								<span
									className={`relative left-0 top-0 my-1 block h-0.5 w-0 rounded-sm bg-black delay-[0] duration-200 ease-in-out ${
										!sidebarOpen && '!w-full delay-300'
									}`}></span>
								<span
									className={`relative left-0 top-0 my-1 block h-0.5 w-0 rounded-sm bg-black delay-150 duration-200 ease-in-out ${
										!sidebarOpen && 'delay-400 !w-full'
									}`}></span>
								<span
									className={`relative left-0 top-0 my-1 block h-0.5 w-0 rounded-sm bg-black delay-200 duration-200 ease-in-out ${
										!sidebarOpen && '!w-full delay-500'
									}`}></span>
							</span>

							<span className='absolute right-0 h-full w-full rotate-45'>
								<span
									className={`absolute left-2.5 top-0 block h-full w-0.5 rounded-sm bg-black delay-300 duration-200 ease-in-out ${
										!sidebarOpen && '!h-0 !delay-[0]'
									}`}></span>
								<span
									className={`delay-400 absolute left-0 top-2.5 block h-0.5 w-full rounded-sm bg-black duration-200 ease-in-out ${
										!sidebarOpen && '!h-0 !delay-200'
									}`}></span>
							</span>
						</span>
					</button>
				</div>

				{/* <!--  Menu Area --> */}
				<div className='2xsm:gap-7 flex ml-[auto] items-center'>
					<div className='flex flex-row items-center gap-x-1 sm:gap-x-6'>
						<ul className='2xsm:gap-4 flex items-center gap-2'>
							{/* <!-- walllet Menu Area --> */}
							<div className='dropdown dropdown-hover dropdown-end '>
								<label
									tabIndex={0}
									className='b-black flex h-10 w-10 cursor-pointer items-center justify-center rounded-[8px]'>
									<IoWallet size={24} color='#48525B' />
								</label>
								<ul
									tabIndex={0}
									className='dropdown-content z-[1] mt-4 menu p-2 shadow=[100px] bg-[#FF8216] rounded-box w-52'>
									<li>
										<h1 className='text-[white] font-semibold'>
											Wallet balance = {user.wallet} $
										</h1>
									</li>
								</ul>
							</div>
							{/* <!-- walllet Menu Area --> */}
						</ul>
					</div>

					<div className=' flex flex-row items-center gap-x-1 sm:gap-x-6 '>
						<ul className='2xsm:gap-4 flex items-center gap-2 '>
							{/* <!-- Notification Menu Area --> */}
							<DropdownNotification />
							{/* <!-- Notification Menu Area --> */}
						</ul>
					</div>
					{/* <!-- User Area --> */}
					<div className='relative'>
						<Link className='flex items-center gap-3' to='#'>
							<span
								className='h-10 w-10 overflow-hidden rounded-full'
								onClick={() => navigate(`/myProfile`)}>
								<img
									src={
										user.picture !== null
											? `${HOSTNAME}users/picture?path=${user.picture}`
											: UserOne
									}
									alt='User'
								/>
							</span>
							<span className='hidden text-right lg:block'>
								<span
									className='block text-sm font-medium text-black'
									onClick={() => navigate(`/myProfile`)}>
									{user.username}
								</span>
							</span>
						</Link>
					</div>
					{/* <!-- User Area --> */}
				</div>
			</div>
		</header>
	);
};

export default Header;
