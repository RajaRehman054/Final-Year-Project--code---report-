import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { MdNotifications } from 'react-icons/md';
import { getNotifications } from '../api/Api';
import { format } from 'timeago.js';

const DropdownNotification = () => {
	const [notify, setNotify] = useState([]);
	const fetchData = async () => {
		const res = await getNotifications();
		setNotify(res);
	};
	useEffect(() => {
		fetchData();
	}, []);

	return (
		<div className='dropdown dropdown-end '>
			<label
				tabIndex={0}
				className='b-black flex h-10 w-10 cursor-pointer items-center justify-center rounded-[8px]'>
				<MdNotifications size={28} color='#48525B' />
			</label>
			<ul
				tabIndex={0}
				className='dropdown-content max-h-[400px] border-stroke mt-4 flex w-72 overflow-y-auto flex-col rounded-lg border bg-white shadow-md sm:w-80 '>
				<div className='px-4 py-3'>
					<h5 className='text-sm font-medium text-heading'>
						Notification
					</h5>
				</div>
				{notify.length !== 0 ? (
					notify.map((element, index) => (
						<li key={index}>
							<Link
								className='flex flex-col gap-2 border-t border-gray-200 px-4 py-3 hover:bg-gray-100'
								to='#'>
								<p className='text-sm'>{element.message}</p>
								<p className='text-xs'>
									{format(element.createdAt)}
								</p>
							</Link>
						</li>
					))
				) : (
					<li>
						<Link
							className='flex flex-col gap-2 border-t border-gray-200 px-4 py-3 hover:bg-gray-100'
							to='#'>
							<p className='text-sm'>No New Notifications</p>
						</Link>
					</li>
				)}
			</ul>
		</div>
	);
};

export default DropdownNotification;
